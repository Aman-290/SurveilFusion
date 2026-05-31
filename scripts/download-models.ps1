$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "models" | Out-Null

Write-Host "Model download placeholder"
Write-Host "Place your fire/smoke detector at models/fire-yolo.pt or set FIRE_MODEL_PATH in .env."
Write-Host "Recommended path: train/export a detector to ONNX or TensorRT for your target hardware."
