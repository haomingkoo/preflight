# Bugfix Requirements Document

## Introduction

The Railway deployment of the Dash/Flask ML workbench application exhibits excessive memory usage, accumulating approximately 13GB of memory over time. The application allows users to upload CSV/Parquet files (up to 50MB, 5M cells, 120MB serialized), perform exploratory data analysis (EDA), and train machine learning models. While the application enforces file size limits and implements disk-based session caching to `/tmp/preflight-data`, it lacks explicit memory cleanup mechanisms. This results in memory accumulation from dataframes, plotly figures, and model training operations that are not released after operations complete, leading to increased hosting costs ($7.06 estimated vs expected lower costs) and potential out-of-memory failures.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN callbacks complete processing dataframes (upload, EDA, correlation, model training) THEN the system retains dataframe objects in memory without explicit cleanup

1.2 WHEN large plotly figures are generated in callbacks (EDA visualizations, correlation heatmaps, confusion matrices, ROC/PR curves) THEN the system retains figure objects in memory without explicit cleanup

1.3 WHEN model training completes with cross-validation on large datasets THEN the system retains intermediate model objects, cross-validation results, and transformed data in memory without explicit cleanup

1.4 WHEN multiple sequential operations are performed within a session THEN the system accumulates memory from previous operations without releasing it

1.5 WHEN stale session data cleanup runs (every 5 minutes for files older than 2 hours) THEN the system only removes disk files but does not trigger Python garbage collection to release associated in-memory objects

### Expected Behavior (Correct)

2.1 WHEN callbacks complete processing dataframes (upload, EDA, correlation, model training) THEN the system SHALL explicitly delete dataframe references and trigger garbage collection to release memory

2.2 WHEN large plotly figures are generated in callbacks THEN the system SHALL explicitly delete intermediate figure objects and trigger garbage collection after serialization

2.3 WHEN model training completes with cross-validation THEN the system SHALL explicitly delete intermediate model objects, cross-validation results, and transformed data after extracting final metrics

2.4 WHEN multiple sequential operations are performed within a session THEN the system SHALL release memory from previous operations through explicit cleanup and garbage collection

2.5 WHEN stale session data cleanup runs THEN the system SHALL trigger Python garbage collection after removing disk files to release associated in-memory objects

### Unchanged Behavior (Regression Prevention)

3.1 WHEN file size limits are enforced (50MB upload, 5M cells, 120MB serialized, 50k training rows) THEN the system SHALL CONTINUE TO reject oversized files with appropriate error messages

3.2 WHEN dataframes are saved to disk-based session cache THEN the system SHALL CONTINUE TO use gzip compression and secure file permissions

3.3 WHEN stale datasets are cleaned up THEN the system SHALL CONTINUE TO remove files older than DATA_RETENTION_SECONDS (2 hours default)

3.4 WHEN callbacks return results to the UI THEN the system SHALL CONTINUE TO provide the same visualizations, metrics, and data outputs

3.5 WHEN rate limiting is applied (8 uploads/min, 12 trains/min) THEN the system SHALL CONTINUE TO enforce these limits per client IP

3.6 WHEN users perform EDA, correlation analysis, or model training THEN the system SHALL CONTINUE TO produce identical results and visualizations
