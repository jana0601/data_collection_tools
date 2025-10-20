#!/usr/bin/env python3
"""
Quick gesture class statistics analyzer.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import GestureDataPreprocessor

def print_gesture_stats():
    """Print gesture class statistics."""
    print("GESTURE CLASS STATISTICS")
    print("=" * 40)
    
    # Initialize preprocessor
    preprocessor = GestureDataPreprocessor("data/records.jsonl")
    
    # Load data
    df = preprocessor.load_data()
    
    if len(df) == 0:
        print("No data found in records.jsonl")
        return
    
    # Get analysis
    analysis = preprocessor.analyze_gesture_patterns(df)
    
    print(f"Total Gesture Classes: {analysis['total_gestures']}")
    print(f"Total Samples: {analysis['total_samples']}")
    print()
    
    print("CLASS BREAKDOWN:")
    print("-" * 40)
    
    # Sort by count (descending)
    sorted_gestures = sorted(
        analysis['gesture_distribution'].items(), 
        key=lambda x: x[1]['count'], 
        reverse=True
    )
    
    for gesture, info in sorted_gestures:
        print(f"{gesture:15} | {info['count']:3d} samples | {info['percentage']:5.1f}%")
    
    print("-" * 40)
    print(f"{'TOTAL':15} | {analysis['total_samples']:3d} samples | 100.0%")

if __name__ == "__main__":
    print_gesture_stats()
