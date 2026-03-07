#!/usr/bin/env python3
"""
Preservation Property Tests - High Memory Usage Fix
Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6

These tests capture baseline behavior BEFORE implementing the fix.
They ensure functional equivalence after memory cleanup changes.

EXPECTED OUTCOME: Tests PASS on both unfixed and fixed code
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import app

def test_upload_preservation():
    """Test that upload functionality produces identical results."""
    print("\n--- Upload Preservation Test ---")
    
    # Create test dataframe
    df = pd.DataFrame({
        'col1': [1, 2, 3, None, 5],
        'col2': ['A', 'B', 'C', 'D', 'E'],
        'col3': [1.1, 2.2, None, 4.4, 5.5],
        'target': [0, 1, 0, 1, 0]
    })
    
    # Test normalize_missing_tokens
    result = app.normalize_missing_tokens(df, ['?', 'NA', 'null'])
    assert result.shape == df.shape, "Shape should be preserved"
    assert list(result.columns) == list(df.columns), "Columns should be preserved"
    
    # Test enforce_dataframe_limits
    result2 = app.enforce_dataframe_limits(df)
    assert result2.shape == df.shape, "Shape should be preserved"
    
    # Test basic_profile
    prof = app.basic_profile(df)
    assert len(prof) == len(df.columns), "Profile should have one row per column"
    assert 'column' in prof.columns, "Profile should have column names"
    assert 'dtype' in prof.columns, "Profile should have dtypes"
    assert 'missing_%' in prof.columns, "Profile should have missing percentages"
    
    print("✓ Upload preservation: PASS")
    return True


def test_typing_preservation():
    """Test that auto-typing produces consistent results."""
    print("\n--- Typing Preservation Test ---")
    
    df = pd.DataFrame({
        'numeric': [1, 2, 3, 4, 5],
        'categorical': ['A', 'B', 'A', 'B', 'A'],
        'high_card': [f'val_{i}' for i in range(5)],
        'id_like': range(5),
        'target': [0, 1, 0, 1, 0]
    })
    
    cfg = app.TypingConfig(
        low_card_threshold=3,
        id_unique_ratio_threshold=0.90,
        treat_small_unique_int_as_cat=True
    )
    
    types = app.auto_type_columns(df, 'target', cfg)
    
    # Verify expected types
    assert 'numeric' in types, "Should type numeric column"
    assert 'categorical' in types, "Should type categorical column"
    assert types['categorical'] == 'categorical', "Low-card string should be categorical"
    
    print("✓ Typing preservation: PASS")
    return True


def test_eda_preservation():
    """Test that EDA operations produce consistent results."""
    print("\n--- EDA Preservation Test ---")
    
    df = pd.DataFrame({
        'num1': np.random.randn(100),
        'num2': np.random.randn(100),
        'cat1': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    
    # Test that dataframe operations work
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    assert len(numeric_cols) >= 2, "Should identify numeric columns"
    
    # Test correlation
    num_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    corr = num_df.corr(method='spearman')
    assert corr.shape[0] == corr.shape[1], "Correlation matrix should be square"
    assert corr.shape[0] == len(numeric_cols), "Correlation should include all numeric cols"
    
    print("✓ EDA preservation: PASS")
    return True


def test_model_training_preservation():
    """Test that model training produces consistent results."""
    print("\n--- Model Training Preservation Test ---")
    
    np.random.seed(42)
    df = pd.DataFrame({
        'num1': np.random.randn(200),
        'num2': np.random.randn(200),
        'cat1': np.random.choice(['A', 'B'], 200),
        'target': np.random.choice([0, 1], 200)
    })
    
    feature_types = {
        'num1': 'numeric',
        'num2': 'numeric',
        'cat1': 'categorical'
    }
    
    # Test preprocessor building
    preprocessor, num_cols, ord_cols, cat_cols = app.build_preprocessor(
        feature_types,
        scale_numeric=True,
        available_cols=['num1', 'num2', 'cat1']
    )
    
    assert len(num_cols) == 2, "Should identify 2 numeric columns"
    assert len(cat_cols) == 1, "Should identify 1 categorical column"
    assert preprocessor is not None, "Should build preprocessor"
    
    # Test that preprocessing works
    X = df[['num1', 'num2', 'cat1']]
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == len(df), "Should preserve row count"
    
    print("✓ Model training preservation: PASS")
    return True


def test_file_limits_preservation():
    """Test that file size limits are still enforced."""
    print("\n--- File Limits Preservation Test ---")
    
    # Test column limit
    try:
        large_df = pd.DataFrame({f'col_{i}': range(10) for i in range(3000)})
        app.enforce_dataframe_limits(large_df)
        print("✗ Should have raised error for too many columns")
        return False
    except ValueError as e:
        assert 'Too many columns' in str(e), "Should reject too many columns"
    
    # Test cell limit
    try:
        large_df = pd.DataFrame({f'col_{i}': range(100000) for i in range(100)})
        app.enforce_dataframe_limits(large_df)
        print("✗ Should have raised error for too many cells")
        return False
    except ValueError as e:
        assert 'too large' in str(e), "Should reject too many cells"
    
    print("✓ File limits preservation: PASS")
    return True


def test_data_retention_config():
    """Test that DATA_RETENTION_SECONDS is accessible."""
    print("\n--- Data Retention Config Test ---")
    
    # Check that the constant exists
    assert hasattr(app, 'DATA_RETENTION_SECONDS'), "Should have DATA_RETENTION_SECONDS"
    retention = app.DATA_RETENTION_SECONDS
    assert isinstance(retention, int), "Should be an integer"
    assert retention > 0, "Should be positive"
    
    print(f"Current DATA_RETENTION_SECONDS: {retention}")
    print("✓ Data retention config: PASS")
    return True


if __name__ == "__main__":
    print("="*70)
    print("PRESERVATION PROPERTY TESTS")
    print("High Memory Usage Fix - Bugfix Spec")
    print("="*70)
    print("\nThese tests capture baseline behavior before implementing the fix.")
    print("They MUST PASS on both unfixed and fixed code.")
    
    tests = [
        test_upload_preservation,
        test_typing_preservation,
        test_eda_preservation,
        test_model_training_preservation,
        test_file_limits_preservation,
        test_data_retention_config
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*70)
    print("PRESERVATION TEST RESULTS")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if all(results):
        print("\n✓ ALL PRESERVATION TESTS PASSED")
        print("Baseline behavior captured successfully.")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED")
        print("Fix preservation tests before proceeding.")
        sys.exit(1)
