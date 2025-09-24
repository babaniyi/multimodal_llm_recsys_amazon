# Configuration Guide

This guide explains the key configuration options in `train_config.json` and their impact on addressing reviewer feedback.

## Key Configuration Sections

### 1. Dataset Scale (Addresses "Scale Limitations" Feedback)

```json
"model_config": {
    "test_size": 100000,                    // Dataset size (was 40K)
    "min_dataset_size": 100000,             // Minimum size for meaningful evaluation
    "max_dataset_size": 500000             // Maximum size for computational limits
}
```

**Impact**: Increases dataset from 40K to 100K+ interactions for more credible evaluation.

### 2. Multi-Category Support (Addresses "Single Domain" Feedback)

```json
"data_config": {
    "use_multiple_categories": false,       // false=single category, true=multiple categories
    "category_balance": false,              // false=natural distribution, true=balanced
    "cross_category_evaluation": false     // false=standard split, true=cross-category testing
}
```

**Options**:
- **Conservative**: All `false` (current) - Single category, original behavior
- **Enhanced**: All `true` - Multiple categories, balanced data, cross-category evaluation

### 3. Statistical Testing (Addresses "Evaluation Credibility" Feedback)

```json
"training_config": {
    "evaluation_runs": 5,                   // Number of runs for statistical testing
    "statistical_testing": true,            // Enable robust evaluation
    "confidence_level": 0.95,              // 95% confidence intervals
    "ablation_studies": true               // Enable modality ablation studies
}
```

**Impact**: Provides statistical significance testing, confidence intervals, and ablation analysis.

### 4. Enhanced Baselines (Addresses "Missing Baselines" Feedback)

The system automatically includes:
- **Traditional**: CF, Content-based, Hybrid, Popularity, Matrix Factorization
- **Multimodal**: VBPR, DeepCoNN, NRMF, SASRec (when enabled)

## Quick Configuration Presets

### Original Workflow (Conservative)
```json
{
    "model_config": {"test_size": 40000, "test_mode": true},
    "data_config": {"use_multiple_categories": false},
    "training_config": {"statistical_testing": false, "ablation_studies": false}
}
```

### Enhanced Workflow (Addresses Reviewer Feedback)
```json
{
    "model_config": {"test_size": 100000, "test_mode": false},
    "data_config": {"use_multiple_categories": true, "category_balance": true},
    "training_config": {"statistical_testing": true, "ablation_studies": true, "evaluation_runs": 5}
}
```

### Research-Grade (Maximum Credibility)
```json
{
    "model_config": {"test_size": 200000, "test_mode": false},
    "data_config": {"use_multiple_categories": true, "category_balance": true, "cross_category_evaluation": true},
    "training_config": {"statistical_testing": true, "ablation_studies": true, "evaluation_runs": 10, "k_folds": 10}
}
```

## Configuration Impact on Reviewer Concerns

| Reviewer Concern | Configuration Solution | Impact |
|------------------|----------------------|---------|
| Perfect scores unrealistic | `"statistical_testing": true` | Multiple runs, confidence intervals |
| Missing multimodal baselines | Automatic inclusion | VBPR, DeepCoNN, NRMF, SASRec |
| No ablation studies | `"ablation_studies": true` | Text-only, image-only, fusion analysis |
| Single domain limitation | `"use_multiple_categories": true` | Multiple Amazon categories |
| Dataset scale too small | `"test_size": 100000` | 100K+ interactions |
| No statistical testing | `"evaluation_runs": 5` | Statistical significance testing |

## Usage Examples

### Enable Multi-Category Evaluation
```json
"data_config": {
    "use_multiple_categories": true,
    "category_balance": true,
    "cross_category_evaluation": true
}
```

### Enable Robust Statistical Testing
```json
"training_config": {
    "statistical_testing": true,
    "evaluation_runs": 5,
    "confidence_level": 0.95,
    "ablation_studies": true
}
```

### Increase Dataset Scale
```json
"model_config": {
    "test_size": 200000,
    "test_mode": false,
    "min_dataset_size": 200000
}
```

## Notes

- All enhanced features are **backward compatible**
- If enhanced features fail, system falls back to original behavior
- Configuration changes take effect on next run
- See `train_config_with_comments.json` for detailed explanations
