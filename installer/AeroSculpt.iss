; Inno Setup script for AeroSculpt Desktop. Compiled by scripts/build_desktop.py --installer:
;   ISCC /DAppVersion=1.1.0 /DSourceDir=dist\AeroSculpt /DOutDir=dist installer\AeroSculpt.iss
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\AeroSculpt"
#endif
#ifndef OutDir
  #define OutDir "..\dist"
#endif

[Setup]
AppId={{7D3B1C5E-5A3F-4F1A-9C7E-AEROSCULPT01}
AppName=AeroSculpt
AppVersion={#AppVersion}
AppVerName=AeroSculpt {#AppVersion}
AppPublisher=AeroSculpt
AppPublisherURL=https://github.com/swishy1104/aerosculpt
DefaultDirName={autopf}\AeroSculpt
DefaultGroupName=AeroSculpt
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutDir}
OutputBaseFilename=AeroSculpt-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\AeroSculpt.exe
LicenseFile={#SourceDir}\THIRD-PARTY-NOTICES.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\AeroSculpt"; Filename: "{app}\AeroSculpt.exe"
Name: "{group}\AeroSculpt self-check"; Filename: "{app}\AeroSculpt.exe"; Parameters: "--check"
Name: "{autodesktop}\AeroSculpt"; Filename: "{app}\AeroSculpt.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AeroSculpt.exe"; Description: "{cm:LaunchProgram,AeroSculpt}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal"
