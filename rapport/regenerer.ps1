param([string]$Python = '')
$ErrorActionPreference = 'Stop'
$dossierRapport = $PSScriptRoot
if (-not $Python) {
    $candidatsRuntime = Get-ChildItem -Path (Join-Path $env:USERPROFILE '.cache\*runtimes\*\dependencies\python\python.exe') -ErrorAction SilentlyContinue
    if ($candidatsRuntime) { $Python = $candidatsRuntime[0].FullName }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
& $Python (Join-Path $dossierRapport 'construire_rapport.py') --figures --controle
if ($LASTEXITCODE -ne 0) { throw 'La génération du rapport a échoué.' }
& $Python (Join-Path $dossierRapport 'verifier_resultats.py')
if ($LASTEXITCODE -ne 0) { throw 'Le contrôle des résultats a échoué.' }
Write-Host 'Rapport et contrôles régénérés.'
