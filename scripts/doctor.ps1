$ErrorActionPreference = "Stop"

Write-Host "SurveilFusion doctor"
Write-Host "Python:" (python --version)
Write-Host "Docker:" (docker --version)

python -m surveilfusion doctor
