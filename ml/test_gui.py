#!/usr/bin/env python3
"""
Minimal GUI test to isolate model loading issues.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QTextEdit
from PySide6.QtCore import QTimer
from inference_app import GestureInferenceEngine

class MinimalTestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Loading Test")
        self.setGeometry(100, 100, 600, 400)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        
        # Create text area for output
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)
        
        # Initialize inference engine
        self.log("Initializing GestureInferenceEngine...")
        self.inference_engine = GestureInferenceEngine()
        self.log(f"Models directory: {self.inference_engine.models_dir}")
        self.log(f"Models directory exists: {self.inference_engine.models_dir.exists()}")
        
        # Load models
        self.log("Attempting to load models...")
        success = self.inference_engine.load_models()
        
        if success:
            self.log("✅ Models loaded successfully!")
            self.log(f"Classes: {', '.join(self.inference_engine.class_names)}")
            self.log(f"Best model: {self.inference_engine.best_model_name}")
        else:
            self.log("❌ Failed to load models!")
            self.log("This indicates a GUI-specific issue")
    
    def log(self, message):
        """Add message to text area."""
        self.text_area.append(message)
        print(message)  # Also print to console

def main():
    app = QApplication(sys.argv)
    window = MinimalTestApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
