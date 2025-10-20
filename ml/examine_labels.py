#!/usr/bin/env python3
"""
Debug script to examine the actual labels in the dataset.
"""

import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def examine_labels():
    """Examine the actual labels in the dataset."""
    data_path = Path("data/records.jsonl")
    
    if not data_path.exists():
        print("No records.jsonl found")
        return
    
    print("EXAMINING DATASET LABELS")
    print("=" * 40)
    
    labels = []
    unique_labels = set()
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                label = record.get('label', 'NO_LABEL')
                labels.append(label)
                unique_labels.add(label)
                
                # Show first few records
                if line_num <= 5:
                    print(f"Record {line_num}: label = '{label}'")
                    
            except json.JSONDecodeError as e:
                print(f"Invalid JSON on line {line_num}: {e}")
    
    print(f"\nSUMMARY:")
    print(f"Total records: {len(labels)}")
    print(f"Unique labels found: {sorted(unique_labels)}")
    print(f"Number of unique labels: {len(unique_labels)}")
    
    # Count each label
    from collections import Counter
    label_counts = Counter(labels)
    print(f"\nLabel counts:")
    for label, count in label_counts.most_common():
        print(f"  '{label}': {count} samples")

if __name__ == "__main__":
    examine_labels()
