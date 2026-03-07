#!/usr/bin/env python3
"""Bug Exploration Test - High Memory Usage Fix"""

import gc
import sys
import tracemalloc
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import app

def get_memory_mb():
    current, peak = tracemalloc.get_traced_memory()
    return current / (1024 * 1024)

def test_memory():
    print("Testing memory accumulation...")
    tracemalloc.start()
    gc.collect()
    baseline = get_memory_mb()
    print(f"Baseline: {baseline:.2f} MB")
    
    # Create test data
    df = pd.DataFrame({
        'a': range(50000),
        'b': range(50000),
        'target': [0, 1] * 25000
    })
    
    # Simulate operations
    for i in range(5):
        prof = app.basic_profile(df)
    
    gc.collect()
    final = get_memory_mb()
    growth = final - baseline
    
    print(f"Final: {final:.2f} MB, Growth: {growth:.2f} MB")
    
    # Check for gc.collect in cleanup
    import inspect
    source = inspect.getsource(app._cleanup_stale_datasets)
    has_gc = 'gc.collect()' in source
    print(f"Has gc.collect(): {has_gc}")
    
    tracemalloc.stop()
    
    if growth > 30 or not has_gc:
        print("FAIL: Bug confirmed")
        return False
    print("PASS: Bug fixed")
    return True

if __name__ == "__main__":
    result = test_memory()
    sys.exit(0 if result else 1)
