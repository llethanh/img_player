; ============================================================================
; Flick Player — Inno Setup script
; ============================================================================
; Wraps the PyInstaller --onedir bundle (``dist/img_player/``) into a
; standard Windows installer:
;   - Start menu shortcut + optional desktop icon
;   - Add/Remove Programs entry
;   - .session file association (double-click → opens in Flick)
;   - Per-user install (no admin required)
;
; PREREQ: build the bundle first with ``build_exe.bat`` so
; ``dist\img_player\img_player.exe`` exists.
;
; HOW TO USE
; ----------
; 1. Install Inno Setup 6+ from https://jrsoftware.org/isinfo.php
; 2. Open this file in Inno Setup Compiler, click "Compile"
;    OR run from CLI:
;      "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\flick.iss
; 3. Output goes to ``installer\Output\flick-setup-X.Y.Z.exe``
;
; CODE SIGNING (when you have a cert)
; -----------------------------------
; Uncomment the SignTool directive below and configure your cert. Without
; signing, Windows SmartScreen will scare users with "unrecognized publisher".
; A standard EV code signing cert is ~$300/year.
;
; FILE ASSOCIATION NOTE
; ---------------------
; The .session association registers under HKCU (per-user) so it doesn't
; collide with anything else. Double-click in Explorer → arg passed to
; img_player.exe → app.py routes via _on_open_session_requested.
; ============================================================================

#define MyAppName "Flick Player"
#define MyAppVersion "1.0.2"
#define MyAppPublisher "llethanh"
#define MyAppURL "https://github.com/llethanh/img_player"
#define MyAppExeName "img_player.exe"
#define MyAppId "{{F41C4FA1-2D7A-4E5B-9FEB-FL1CKPLAYER001}"

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\Flick Player
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=Output
OutputBaseFilename=flick-setup-{#MyAppVersion}
SetupIconFile=
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Per-user install — no admin elevation needed. Studios with locked-down
; machines can install without IT involvement.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64
; Uninstaller cosmetics
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName} {#MyAppVersion}

; ----- Code signing (uncomment once you have a cert) -----
; SignTool=signtool sign /f "C:\path\to\cert.pfx" /p "PASSWORD" /tr http://timestamp.digicert.com /td sha256 /fd sha256 $f
; SignedUninstaller=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "associatesession"; Description: "Associate &.session files with {#MyAppName}"; GroupDescription: "File associations:"; Flags: checkedonce

[Files]
; Bundle the entire PyInstaller --onedir output. ``recursesubdirs`` walks
; the _internal/ tree (Qt6, OIIO, OCIO, FFmpeg, all the bundled DLLs +
; data); ``createallsubdirs`` mirrors the structure under {app}.
Source: "..\dist\img_player\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; .session double-click association. HKCU keeps it per-user; no admin needed.
Root: HKCU; Subkey: "Software\Classes\.session"; ValueType: string; ValueData: "FlickPlayer.session"; Flags: uninsdeletevalue; Tasks: associatesession
Root: HKCU; Subkey: "Software\Classes\FlickPlayer.session"; ValueType: string; ValueData: "Flick Session"; Flags: uninsdeletekey; Tasks: associatesession
Root: HKCU; Subkey: "Software\Classes\FlickPlayer.session\DefaultIcon"; ValueType: string; ValueData: "{app}\{#MyAppExeName},0"; Tasks: associatesession
Root: HKCU; Subkey: "Software\Classes\FlickPlayer.session\shell\open\command"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: associatesession

[Run]
; Offer to launch right after install. ``nowait`` so the installer wizard
; closes cleanly even if the app takes a moment to spin up.
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
var
  ErrorCode: Integer;
begin
  // Sanity: refuse to install on Windows < 10. PySide6 / Qt6 won't run
  // on anything older anyway.
  if (GetWindowsVersion shr 24) < 10 then
  begin
    MsgBox('Flick Player requires Windows 10 or later.', mbCriticalError, MB_OK);
    Result := False;
    exit;
  end;
  Result := True;
end;
