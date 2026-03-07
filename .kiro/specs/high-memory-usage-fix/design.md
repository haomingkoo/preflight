# High Memory Usage Fix - Bugfix Design

## Overview

The Railway deployment accumulates approximately 13GB of memory over time due to lack of explicit memory cleanup after dataframe operations, figure generation, and model training. The fix implements explicit garbage collection at strategic points in callback functions and reduces DATA_RETENTION_SECONDS from 7200 (2 hours) to 1800 (30 minutes) for faster cleanup of inactive sessions. This targeted approach releases memory from completed operations without changing application functionality or user-visible behavior.

## Glossary

- **Bug_Condition (C)**: The condition that triggers excessive memory accumulation - when callbacks complete operations but retain large objects (dataframes, figures, models) in memory without explicit cleanup
- **Property (P)**: The desired behavior - memory should be released after callback operations complete through explicit deletion and garbage collection
- **Preservation**: All existing functionality (file limits, visualizations, metrics, rate limiting, caching) must remain unchanged
- **Callback**: Dash callback functions that process user interactions (upload, EDA, correlation, model training)
- **gc.collect()**: Python's explicit garbage collection function that forces immediate memory reclamation
- **DATA_RETENTION_SECONDS**: Configuration constant controlling how long session data persists on disk (currently 7200 seconds / 2 hours)
- **save_df_for_session**: Function in `app.py` that serializes dataframes to disk cache at `/tmp/preflight-data`
- **\_cleanup_stale_datasets**: Function in `app.py` that removes disk files older than DATA_RETENTION_SECONDS (runs every 5 minutes)

## Bug Details

### Bug Condition

The bug manifests when callbacks complete processing large dataframes, generate plotly figures, or train machine learning models. The Python garbage collector does not immediately release memory for these objects, causing accumulation over multiple operations within and across sessions.

**Formal Specification:**

```
FUNCTION isBugCondition(operation)
  INPUT: operation of type CallbackOperation
  OUTPUT: boolean

  RETURN operation.type IN ['upload', 'eda', 'correlation', 'model_training']
         AND operation.completed = true
         AND operation.largeObjectsInScope = true
         AND NOT explicitMemoryCleanupPerformed(operation)
END FUNCTION
```

### Examples

- **Upload Callback**: User uploads 50MB CSV (300k rows, 50 columns). After `on_upload` callback completes and returns dataset token, the parsed dataframe remains in memory even though it's saved to disk cache.

- **EDA Multi-Plot**: User generates 20 EDA plots for different features. After `update_eda_multi` callback completes, all intermediate dataframes (filtered, sampled) and plotly figure objects remain in memory.

- **Correlation Analysis**: User views correlation heatmap with 100 numeric features. After `update_corr` callback completes, the correlation matrix dataframe and scatter plot figures remain in memory.

- **Model Training**: User trains RandomForest on 50k rows with 5-fold cross-validation. After `train_model` callback completes, intermediate objects remain in memory: X_train, X_test, y_train, y_test, cv_scores, fitted pipeline, probability arrays, confusion matrix, ROC/PR curve data.

- **Stale Session Cleanup**: Every 5 minutes, `_cleanup_stale_datasets` removes disk files older than 2 hours but does not trigger garbage collection, leaving associated in-memory references uncollected.

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**

- File size limits (50MB upload, 5M cells, 120MB serialized, 50k training rows) must continue to reject oversized files with appropriate error messages
- Disk-based session caching must continue to use gzip compression and secure file permissions
- Rate limiting (8 uploads/min, 12 trains/min) must continue to enforce limits per client IP
- All visualizations, metrics, and data outputs must remain identical
- EDA plots, correlation heatmaps, confusion matrices, ROC/PR curves must display exactly as before
- Model training results (classification reports, cross-validation scores) must be numerically identical
- Session management and authentication behavior must remain unchanged

**Scope:**
All inputs and operations should produce identical user-visible results. The only change is internal memory management - releasing objects after they're no longer needed. This includes:

- All callback return values must be identical
- All UI components must display the same content
- All error messages and validation logic must remain unchanged
- All disk caching behavior must remain unchanged

## Hypothesized Root Cause

Based on the bug description and codebase analysis, the root causes are:

1. **No Explicit Cleanup in Callbacks**: Callbacks like `on_upload`, `update_eda_multi`, `update_corr`, and `train_model` create large objects (dataframes, figures, models) but never explicitly delete them or call `gc.collect()`. Python's garbage collector runs periodically but may not reclaim memory quickly enough under continuous load.

2. **Long Data Retention Period**: DATA_RETENTION_SECONDS is set to 7200 (2 hours), meaning inactive session data persists for a long time. While disk files are cleaned up, the associated in-memory objects may linger if garbage collection hasn't run.

3. **No GC After Stale Cleanup**: The `_cleanup_stale_datasets` function removes disk files but doesn't call `gc.collect()` afterward, missing an opportunity to reclaim memory from deleted session data.

4. **Accumulation Across Operations**: Users performing multiple sequential operations (upload → EDA → correlation → training) accumulate memory from each step without intermediate cleanup.

5. **Large Intermediate Objects**: Model training creates many intermediate objects (train/test splits, cross-validation results, probability arrays, transformed data) that remain in scope after the callback completes.

## Correctness Properties

Property 1: Bug Condition - Memory Released After Callback Completion

_For any_ callback operation that processes large objects (dataframes, figures, models), the fixed code SHALL explicitly delete local references to large objects and call gc.collect() before returning, ensuring memory is released promptly after the operation completes.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

Property 2: Preservation - Identical Functional Behavior

_For any_ user interaction with the application (upload, EDA, correlation, training), the fixed code SHALL produce exactly the same outputs, visualizations, metrics, and error messages as the original code, preserving all existing functionality.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `app.py`

**Specific Changes**:

1. **Import gc Module**: Add `import gc` at the top of the file with other imports

2. **Reduce DATA_RETENTION_SECONDS**: Change from 7200 (2 hours) to 1800 (30 minutes)

   - Line ~79: `DATA_RETENTION_SECONDS = int(os.getenv("PREFLIGHT_DATA_RETENTION_SECONDS", "1800"))`
   - Rationale: Faster cleanup of inactive session data reduces memory footprint

3. **Add GC to Stale Cleanup Function**: In `_cleanup_stale_datasets` (line ~192)

   - After the cleanup loop completes, add `gc.collect()` to reclaim memory from deleted sessions
   - This ensures memory is released when disk files are removed

4. **Add GC to Upload Callback**: In `on_upload` function (line ~1359)

   - Before the final return statement, add explicit cleanup:
     ```python
     del df  # if not already out of scope
     gc.collect()
     ```
   - This releases memory from the parsed dataframe after it's saved to disk

5. **Add GC to EDA Multi Callback**: In `update_eda_multi` function (line ~1681)

   - Before the final return statement, add cleanup for intermediate dataframes and figures:
     ```python
     del df, tmp_df  # and any other large intermediate objects
     gc.collect()
     ```

6. **Add GC to Correlation Callback**: In `update_corr` function (line ~1785)

   - Before the final return statement, add cleanup:
     ```python
     del df, num_df, corr  # and scatter plot dataframes
     gc.collect()
     ```

7. **Add GC to Model Training Callback**: In `train_model` function (line ~1928)

   - Before the final return statement, add comprehensive cleanup:
     ```python
     del df, X, y_raw, X_train, X_test, y_train, y_test, pipe, cv_scores, y_pred, y_proba, proba, cm
     gc.collect()
     ```
   - This is the most memory-intensive callback and will benefit most from explicit cleanup

8. **Add GC to Health Update Callback**: In `update_health` function (line ~1435)
   - Before the final return statement, add cleanup:
     ```python
     del df, prof
     gc.collect()
     ```

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, demonstrate the memory accumulation bug on unfixed code through memory profiling, then verify the fix reduces memory usage while preserving all functional behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface evidence that demonstrates the bug BEFORE implementing the fix. Measure memory accumulation across multiple operations to confirm the root cause analysis.

**Test Plan**: Write memory profiling tests that simulate realistic user workflows (upload → EDA → correlation → training) and measure memory usage before and after each operation. Run these tests on the UNFIXED code to observe memory accumulation patterns.

**Test Cases**:

1. **Upload Memory Test**: Upload a 40MB CSV, measure memory before/after callback (will show memory retained on unfixed code)
2. **EDA Memory Test**: Generate 20 EDA plots, measure memory growth (will show accumulation on unfixed code)
3. **Correlation Memory Test**: Generate correlation heatmap with 100 features, measure memory (will show figure retention on unfixed code)
4. **Training Memory Test**: Train RandomForest with 5-fold CV on 50k rows, measure memory (will show largest accumulation on unfixed code)
5. **Sequential Operations Test**: Perform upload → EDA → correlation → training in sequence, measure cumulative memory growth (will show additive accumulation on unfixed code)
6. **Stale Cleanup Test**: Trigger stale dataset cleanup, measure if memory is released (will show no GC on unfixed code)

**Expected Counterexamples**:

- Memory usage increases after each callback and does not decrease
- Memory remains elevated even after operations complete
- Stale cleanup removes disk files but memory usage stays high
- Possible causes: no explicit deletion, no gc.collect() calls, long retention period

### Fix Checking

**Goal**: Verify that for all callback operations where the bug condition holds, the fixed code releases memory promptly after completion.

**Pseudocode:**

```
FOR ALL operation WHERE isBugCondition(operation) DO
  memory_before := getCurrentMemoryUsage()
  result := executeCallback_fixed(operation)
  memory_after := getCurrentMemoryUsage()
  ASSERT memory_after < memory_before + threshold
  ASSERT result is functionally correct
END FOR
```

**Testing Approach**: Use memory profiling tools (memory_profiler, tracemalloc, or psutil) to measure memory usage before and after each callback. Verify that memory decreases or stays bounded after gc.collect() is called.

### Preservation Checking

**Goal**: Verify that for all user interactions, the fixed code produces exactly the same outputs as the original code.

**Pseudocode:**

```
FOR ALL user_interaction IN [upload, eda, correlation, training] DO
  result_original := executeCallback_original(user_interaction)
  result_fixed := executeCallback_fixed(user_interaction)
  ASSERT result_original = result_fixed
  ASSERT visualizations are identical
  ASSERT metrics are numerically identical
END FOR
```

**Testing Approach**: Snapshot testing is recommended for preservation checking because:

- It captures exact outputs (dataframes, figures, metrics) from the original code
- It detects any unintended changes in callback return values
- It provides strong guarantees that user-visible behavior is unchanged

**Test Plan**: Run all callbacks on UNFIXED code with known inputs, capture outputs (dataset tokens, figure JSON, metrics), then run the same inputs on FIXED code and compare outputs byte-for-byte.

**Test Cases**:

1. **Upload Preservation**: Verify uploaded dataframe content, dataset token, and success message are identical
2. **EDA Preservation**: Verify all EDA plots (histograms, box plots, scatter plots) are visually and data-wise identical
3. **Correlation Preservation**: Verify correlation matrix values and heatmap figure are identical
4. **Training Preservation**: Verify classification report, confusion matrix, ROC/PR curves, and cross-validation scores are numerically identical (within floating-point tolerance)
5. **Error Handling Preservation**: Verify all error messages and validation logic produce identical results
6. **Rate Limiting Preservation**: Verify rate limiting behavior is unchanged

### Unit Tests

- Test that gc.collect() is called in each modified callback
- Test that DATA_RETENTION_SECONDS is reduced to 1800
- Test that \_cleanup_stale_datasets calls gc.collect() after cleanup
- Test that all callbacks still return expected data structures
- Test edge cases (empty dataframes, single-row datasets, missing values)

### Property-Based Tests

- Generate random dataframes with varying sizes and verify memory is released after upload
- Generate random feature selections and verify EDA memory cleanup works across configurations
- Generate random model configurations and verify training memory cleanup works for all model types
- Test that memory cleanup works correctly for both binary and multiclass classification

### Integration Tests

- Test full user workflow: upload → EDA → correlation → training, verify memory stays bounded
- Test multiple sequential uploads, verify memory doesn't accumulate linearly
- Test session expiration and cleanup, verify memory is released when sessions expire
- Test concurrent users, verify memory cleanup works correctly under load
- Test that Railway deployment memory usage stays below expected threshold over 24-hour period
