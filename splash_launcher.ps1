# Flick Player splash launcher.
#
# Boots a tiny WPF window with the splash PNG, spawns FlickPlayer.exe,
# then polls for a marker file. When Flick's MainWindow shows it
# writes %TEMP%\flick_ready.flag (see splash.close in src/img_player/
# splash.py) and this launcher closes its window.
#
# Why this exists: PyInstaller's built-in Splash() uses Tcl/Tk, which
# is not DPI-aware on Windows; the splash visibly shrinks and shifts
# ~100 ms after the bootloader paints it. WPF inherits the .NET
# DPI-awareness model so the artwork stays exactly where it lands.
# Compared to a Qt-only splash, PowerShell + WPF starts in ~200 ms
# (PySide6 alone takes ~2 s), so the user sees the splash within a
# perceptual frame of double-clicking.

[void][System.Reflection.Assembly]::LoadWithPartialName('PresentationFramework')
[void][System.Reflection.Assembly]::LoadWithPartialName('PresentationCore')
[void][System.Reflection.Assembly]::LoadWithPartialName('WindowsBase')

$ErrorActionPreference = 'Continue'

# Resolve sibling paths from this script's location so the bundle
# can sit anywhere on disk without us hard-coding paths.
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$exePath   = Join-Path $scriptDir 'FlickPlayer.exe'
$splashPng = Join-Path $scriptDir '_internal\img_player\assets\splash.png'
$markerPath = Join-Path $env:TEMP 'flick_ready.flag'

# Drop a stale marker from a previous run so we don't close the splash
# the moment we open it.
if (Test-Path $markerPath) { Remove-Item $markerPath -Force -ErrorAction SilentlyContinue }

# WPF window: borderless, transparent background, sized to the PNG,
# centred on the primary screen, always-on-top, no taskbar entry.
# SizeToContent + a raw <Image> means the window auto-fits the PNG
# without us computing dimensions.
$xamlString = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        WindowStyle="None"
        AllowsTransparency="True"
        Background="Transparent"
        ResizeMode="NoResize"
        ShowInTaskbar="False"
        Topmost="True"
        Focusable="False"
        WindowStartupLocation="CenterScreen"
        Width="480"
        Height="260">
    <Image x:Name="SplashImage" Stretch="Uniform" />
</Window>
"@

[xml]$xaml = $xamlString
$reader = New-Object System.Xml.XmlNodeReader $xaml
try {
    $window = [Windows.Markup.XamlReader]::Load($reader)
} catch {
    # If WPF init failed for any reason, just spawn Flick anyway —
    # the user gets the legacy "no splash" experience instead of a
    # broken launcher.
    Start-Process -FilePath $exePath
    exit 0
}

# Load the PNG via a BitmapImage so the file isn't kept locked on
# disk after the window opens (Image.Source = path string would
# hold a file handle until GC).
if (Test-Path $splashPng) {
    $bitmap = New-Object System.Windows.Media.Imaging.BitmapImage
    $bitmap.BeginInit()
    $bitmap.UriSource = New-Object System.Uri($splashPng, [System.UriKind]::Absolute)
    $bitmap.CacheOption = [System.Windows.Media.Imaging.BitmapCacheOption]::OnLoad
    $bitmap.EndInit()
    $bitmap.Freeze()
    $window.FindName('SplashImage').Source = $bitmap
    # Resize the window to match the bitmap's natural dimensions —
    # the XAML hard-codes 480×260 as a sane default, but if the
    # baked-in PNG ever changes shape the window follows along.
    # WPF measures in DIPs; bitmap pixels at 96 DPI map 1:1, and
    # the OS DPI scaling is applied automatically by WPF on render.
    $window.Width  = $bitmap.PixelWidth
    $window.Height = $bitmap.PixelHeight
}

# Tell Flick we launched it via the wrapper so its own splash code
# stays out of the way (no QSplashScreen overlap, write the ready
# marker on MainWindow show, etc.).
[System.Environment]::SetEnvironmentVariable('FLICK_LAUNCHER', '1', 'Process')
$flickProc = Start-Process -FilePath $exePath -PassThru -WindowStyle Hidden

# Poll the marker every 100 ms. Two close conditions:
#   * marker appeared → Flick is ready, close splash cleanly
#   * Flick exited before signalling → crashed, close splash so the
#     launcher itself doesn't hang forever
$timer = New-Object System.Windows.Threading.DispatcherTimer
$timer.Interval = [TimeSpan]::FromMilliseconds(100)
$timer.Add_Tick({
    if (Test-Path $markerPath) {
        Remove-Item $markerPath -Force -ErrorAction SilentlyContinue
        $timer.Stop()
        $window.Close()
    } elseif ($flickProc.HasExited) {
        $timer.Stop()
        $window.Close()
    }
})
$timer.Start()

# Hard cap: if Flick takes longer than 30 s to signal, close the
# splash anyway. Whatever's blocking startup should have surfaced
# its own UI by then; staying covered by the splash would just
# confuse the user.
$timeoutTimer = New-Object System.Windows.Threading.DispatcherTimer
$timeoutTimer.Interval = [TimeSpan]::FromSeconds(30)
$timeoutTimer.Add_Tick({
    $timeoutTimer.Stop()
    $timer.Stop()
    $window.Close()
})
$timeoutTimer.Start()

# Blocks until $window.Close fires.
$window.ShowDialog() | Out-Null
