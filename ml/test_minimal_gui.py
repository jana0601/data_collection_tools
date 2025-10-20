#!/usr/bin/env python3
"""
Minimal GUI test to check if PySide6 works at all.
"""

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PySide6.QtCore import QTimer

class MinimalWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minimal Test Window")
        self.setGeometry(100, 100, 400, 200)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        
        # Add label
        label = QLabel("If you can see this, PySide6 GUI is working!")
        label.setStyleSheet("font-size: 16px; padding: 20px;")
        layout.addWidget(label)
        
        # Auto-close after 5 seconds
        QTimer.singleShot(5000, self.close)
        
        print("Minimal GUI window created - should be visible now")

def main():
    print("Testing minimal PySide6 GUI...")
    app = QApplication(sys.argv)
    
    window = MinimalWindow()
    window.show()
    
    print("Window.show() called - checking if window is visible...")
    print(f"Window visible: {window.isVisible()}")
    print(f"Window geometry: {window.geometry()}")
    
    # Force window to front
    window.raise_()
    window.activateWindow()
    
    print("Starting GUI event loop...")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
