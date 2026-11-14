# AfterShock: full training run on a GPU machine (Windows, NVIDIA GPU).
# Run from the AfterShock folder:  powershell -ExecutionPolicy Bypass -File scripts\gpu_train_all.ps1
# Output to copy back: the whole runs\ folder and data\aftershock_full.db
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:AFTERSHOCK_DB = "sqlite:///$((Resolve-Path data).Path -replace '\\','/')/aftershock_full.db"
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force runs | Out-Null
$log = "runs\gpu_train_all.log"
function Step($title, [scriptblock]$cmd) {
    "`n===== $title  ($(Get-Date -Format 'HH:mm:ss')) =====" | Tee-Object -FilePath $log -Append
    & $cmd 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $title" }
}

# 1. Environment: CUDA PyTorch + project
if (-not (Test-Path .venv)) { python -m venv .venv }
$py = ".venv\Scripts\python.exe"
Step "install" {
    & $py -m pip install -q --upgrade pip
    & $py -m pip install -q torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
    & $py -m pip install -q -r requirements.txt
    & $py -m pip install -q -e .
}
Step "gpu check" { & $py -c "import torch; assert torch.cuda.is_available(), 'NO CUDA GPU'; print(torch.cuda.get_device_name(0))" }

# 2. Ingest ALL disasters present in data\xbd\train (bigger dataset than the laptop run)
Step "ingest" { & $py -m aftershock.ingest --xbd data\xbd\train }

# 3. Quick sanity run (catches problems in ~2 minutes before the long runs)
Step "sanity" { & $py -m aftershock.train --name sanity --arch siamese --loss focal --epochs 1 --limit 500 --batch-size 64 --num-workers 4 }

# 4. The three models compared in the report (same loss for all, so only the architecture differs)
Step "train post_focal"        { & $py -m aftershock.train --name post_focal        --arch post       --loss focal --epochs 15 --batch-size 128 --num-workers 6 }
Step "train siamese_focal"     { & $py -m aftershock.train --name siamese_focal     --arch siamese    --loss focal --epochs 15 --batch-size 128 --num-workers 6 }
Step "train siamese_fm_focal"  { & $py -m aftershock.train --name siamese_fm_focal  --arch siamese_fm --loss focal --epochs 15 --batch-size 128 --num-workers 6 }

# Pick the final model by VALIDATION macro-F1 (never by test score)
$best = "post_focal","siamese_focal","siamese_fm_focal" | Sort-Object { -(Get-Content "runs\$_\metrics.json" -Raw | ConvertFrom-Json).best_val_macro_f1 } | Select-Object -First 1
"Best model by validation macro-F1: $best" | Tee-Object -FilePath $log -Append

# 5. Uncertainty, explanations, hotspots, results table
Step "predict (MC Dropout)" { & $py -m aftershock.predict --ckpt "runs\$best\best.pt" --passes 20 }
Step "explain (Grad-CAM)"   { & $py -m aftershock.explain --ckpt "runs\$best\best.pt" --most-uncertain 30 }
Step "hotspots (DBSCAN)"    { & $py -m aftershock.hotspots --model-version $best }
Step "results table"        { & $py scripts\results_table.py post_focal siamese_focal siamese_fm_focal }

"`nALL DONE $(Get-Date -Format 'HH:mm:ss'). Copy back: runs\  and  data\aftershock_full.db" | Tee-Object -FilePath $log -Append
