# Implementation Plan

- [ ] 1. Write bug condition exploration test

  - **Property 1: Bug Condition** - Memory Accumulation Across Callback Operations
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate memory accumulation without explicit cleanup
  - **Scoped PBT Approach**: Test memory usage patterns across realistic callback operations (upload, EDA, correlation, training)
  - Test that memory increases after callbacks complete and is NOT released on unfixed code
  - Measure memory before/after each callback operation: upload 40MB CSV, generate 20 EDA plots, correlation heatmap with 100 features, train RandomForest with 5-fold CV
  - Test sequential operations (upload → EDA → correlation → training) to demonstrate cumulative accumulation
  - Test that stale cleanup removes disk files but does NOT trigger garbage collection
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (memory accumulates without cleanup - this proves the bug exists)
  - Document counterexamples found: specific memory growth patterns, operations with highest retention, lack of GC after cleanup
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 2. Write preservation property tests (BEFORE implementing fix)

  - **Property 2: Preservation** - Identical Functional Behavior
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for all callback operations
  - Capture exact outputs: dataset tokens, figure JSON, metrics, error messages
  - Write property-based tests for functional equivalence across all callbacks:
    - Upload: verify dataframe content, dataset token, success message
    - EDA: verify all plot types (histograms, box plots, scatter plots) produce identical figures
    - Correlation: verify correlation matrix values and heatmap figure
    - Training: verify classification report, confusion matrix, ROC/PR curves, CV scores (within floating-point tolerance)
    - Error handling: verify validation logic and error messages
    - Rate limiting: verify rate limit enforcement behavior
  - Property-based testing generates test cases across input variations (different file sizes, feature counts, model types)
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 3. Fix for high memory usage accumulation

  - [ ] 3.1 Add gc module import and reduce data retention period

    - Add `import gc` at the top of `app.py` with other imports
    - Change DATA_RETENTION_SECONDS from 7200 to 1800 (line ~79)
    - Update to: `DATA_RETENTION_SECONDS = int(os.getenv("PREFLIGHT_DATA_RETENTION_SECONDS", "1800"))`
    - _Bug_Condition: isBugCondition(operation) where operation completes but largeObjectsInScope=true and NOT explicitMemoryCleanupPerformed_
    - _Expected_Behavior: Memory released after callback completion through explicit deletion and gc.collect()_
    - _Preservation: All functional behavior (file limits, visualizations, metrics, rate limiting, caching) unchanged_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ] 3.2 Add memory cleanup to \_cleanup_stale_datasets function

    - In `_cleanup_stale_datasets` function (line ~192), after cleanup loop completes
    - Add `gc.collect()` to reclaim memory from deleted sessions
    - Add logging: log number of files deleted and memory usage before/after cleanup
    - _Bug_Condition: Stale cleanup removes disk files but does not trigger GC_
    - _Expected_Behavior: Memory released when disk files are removed_
    - _Preservation: Cleanup behavior unchanged, only adds memory reclamation_
    - _Requirements: 2.5, 3.1_

  - [ ] 3.3 Add memory cleanup to on_upload callback

    - In `on_upload` function (line ~1359), before final return statement
    - Add explicit cleanup: `del df` (if not already out of scope), then `gc.collect()`
    - Add logging: log memory usage before/after cleanup, log dataframe size (rows, columns, memory footprint)
    - _Bug_Condition: Upload callback retains parsed dataframe after saving to disk_
    - _Expected_Behavior: Memory released after dataframe saved to disk cache_
    - _Preservation: Upload functionality, file limits, error handling unchanged_
    - _Requirements: 2.1, 3.1, 3.2_

  - [ ] 3.4 Add memory cleanup to update_health callback

    - In `update_health` function (line ~1435), before final return statement
    - Add cleanup: `del df, prof`, then `gc.collect()`
    - Add logging: log memory usage before/after cleanup, log profiling object size
    - _Bug_Condition: Health callback retains dataframe and profiling objects_
    - _Expected_Behavior: Memory released after health metrics computed_
    - _Preservation: Health display functionality unchanged_
    - _Requirements: 2.2, 3.1, 3.3_

  - [ ] 3.5 Add memory cleanup to update_eda_multi callback

    - In `update_eda_multi` function (line ~1681), before final return statement
    - Add cleanup: `del df, tmp_df` and any other large intermediate objects, then `gc.collect()`
    - Add logging: log memory usage before/after cleanup, log number of plots generated, log intermediate dataframe sizes
    - _Bug_Condition: EDA callback retains intermediate dataframes and plotly figures_
    - _Expected_Behavior: Memory released after plots generated and returned_
    - _Preservation: EDA visualizations, plot types, data sampling unchanged_
    - _Requirements: 2.2, 3.1, 3.3_

  - [ ] 3.6 Add memory cleanup to update_corr callback

    - In `update_corr` function (line ~1785), before final return statement
    - Add cleanup: `del df, num_df, corr` and scatter plot dataframes, then `gc.collect()`
    - Add logging: log memory usage before/after cleanup, log correlation matrix size, log number of features analyzed
    - _Bug_Condition: Correlation callback retains correlation matrix and scatter plot data_
    - _Expected_Behavior: Memory released after correlation analysis complete_
    - _Preservation: Correlation heatmap, scatter plots, numeric feature handling unchanged_
    - _Requirements: 2.3, 3.1, 3.4_

  - [ ] 3.7 Add memory cleanup to train_model callback

    - In `train_model` function (line ~1928), before final return statement
    - Add comprehensive cleanup: `del df, X, y_raw, X_train, X_test, y_train, y_test, pipe, cv_scores, y_pred, y_proba, proba, cm`, then `gc.collect()`
    - Add logging: log memory usage before/after cleanup, log training dataset size, log model size, log number of CV folds, log large object allocations/deallocations
    - This is the most memory-intensive callback and will benefit most from explicit cleanup
    - _Bug_Condition: Training callback retains all intermediate objects (splits, CV results, predictions, model)_
    - _Expected_Behavior: Memory released after model training and metrics computed_
    - _Preservation: Model training results, classification reports, confusion matrices, ROC/PR curves numerically identical_
    - _Requirements: 2.4, 3.1, 3.5_

  - [ ] 3.8 Add centralized memory logging utility

    - Create helper function to log memory usage consistently across all callbacks
    - Function should log: current memory usage, memory delta, timestamp, operation name
    - Include garbage collection trigger logging: log when gc.collect() is called and how much memory was reclaimed
    - This supports future troubleshooting by providing visibility into memory patterns
    - _Preservation: No functional changes, logging only_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ] 3.9 Verify bug condition exploration test now passes

    - **Property 1: Expected Behavior** - Memory Released After Callback Completion
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms memory is released after callbacks complete
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed - memory is now released)
    - Verify memory usage decreases or stays bounded after each callback
    - Verify gc.collect() is called in all modified callbacks
    - Verify logging shows memory cleanup events
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ] 3.10 Verify preservation tests still pass
    - **Property 2: Preservation** - Identical Functional Behavior
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all callback outputs are identical: dataset tokens, figures, metrics, error messages
    - Confirm all visualizations are unchanged: EDA plots, correlation heatmaps, confusion matrices, ROC/PR curves
    - Confirm all validation logic and rate limiting behavior unchanged
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise
  - Verify memory usage stays bounded over multiple operations
  - Verify logging provides visibility into memory cleanup events
  - Verify no functional regressions in any callback
