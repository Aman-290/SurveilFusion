$ErrorActionPreference = "Stop"

Write-Host "SurveilFusion doctor"
Write-Host "Python:" (python --version)
Write-Host "Docker:" (docker --version)

if (-not (Test-Path ".env")) {
  Write-Host "Missing .env. Copy .env.example to .env and edit camera/notification settings."
}

if (-not (Test-Path "config/cameras.example.yml")) {
  throw "Missing config/cameras.example.yml"
}

Write-Host "OK"
