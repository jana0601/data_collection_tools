Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

if (!(Test-Path .\.venv\Scripts\Activate.ps1)) {
    python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install PySide6 opencv-python mediapipe numpy

$env:PYTHONPATH = (Get-Location).Path
python -c "import desktop.app as da; da.main()"


