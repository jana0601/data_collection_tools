# Data collection tool

A tool for capturing and preprocessing camera data, designed to support both static and dynamic gesture recognition. Enables real-time data collection and analysis for use in gesture-controlled interfaces, human-computer interaction, and computer vision applications.

## Interface

![Interface](https://github.com/jana0601/data_collection_tools/blob/main/interface.jpg)


## Prerequisites
- Windows 10/11
- Python 3.10+

## Quick Start (Windows PowerShell)
```powershell
Set-Location E:\Fromyourpath
./run_desktop.ps1
or 
powershell -ExecutionPolicy Bypass -File .\run_desktop.ps1

```
This script creates a virtual environment, installs dependencies, and launches the app.

## Manual Run
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install PySide6 opencv-python mediapipe numpy
$env:PYTHONPATH = (Get-Location).Path
python -c "import desktop.app as da; da.main()"
```

## Features
- Live webcam capture
- MediaPipe Pose estimation
- Landmark and skeleton overlay


## Project Structure
```
data_collection_tools/
  desktop/
    app.py           # PySide6 window, video loop, feedback display
    pose_pipeline.py # MediaPipe Pose wrapper
    overlay.py       # Canvas overlay drawing
    geometry.py      # Angles and normalization utilities
    rules.py         # Tadasana rules and cues
  web/               # (Initial web scaffold; not used in desktop mode)
  run_desktop.ps1    # One-command launcher for Windows
```

## Notes
- All inference is on-device; no video frames are uploaded.
- If the webcam does not open, try changing the camera index in `desktop/app.py`.
