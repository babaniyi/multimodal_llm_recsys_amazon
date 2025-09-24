# Cross-Category Performance Analysis

This document explains the cross-category evaluation system designed to compare model performance across different product categories (Appliances, Digital_Music, Gift_Cards, Health_and_Personal_Care).

## Overview

The cross-category analysis addresses the goal of comparing model performance across various product categories to demonstrate:
- **Model Generalizability**: How well the model performs across different product domains
- **Category-Specific Performance**: Which categories the model excels in and struggles with
- **Baseline Comparisons**: How different baselines perform across categories
- **Performance Variance**: Understanding the consistency of model performance

## Categories Analyzed

1. **Appliances**: Home appliances, kitchen equipment, etc.
2. **Digital_Music**: Digital music downloads, streaming content
3. **Gift_Cards**: Gift cards and vouchers
4. **Health_and_Personal_Care**: Health products, personal care items

## Features

### 1. Category-Specific Evaluation
- Evaluates model performance on each category separately
- Loads category-specific test data
- Calculates metrics (HR@10, MRR, NDCG@10, Precision@10) per category

### 2. Baseline Comparison
- Compares multiple baselines across categories:
  - Collaborative Filtering
  - Content-Based Recommender
  - Popularity Recommender
  - Matrix Factorization
- Identifies best-performing baseline for each category

### 3. Performance Analysis
- **Best/Worst Categories**: Identifies categories where model performs best/worst
- **Performance Variance**: Calculates mean, std, min, max across categories
- **Statistical Insights**: Provides recommendations based on variance analysis

### 4. Visualizations
- **Category Performance Comparison**: Bar charts showing metrics across categories
- **Baseline Performance Comparison**: Multi-category baseline comparison
- **Performance Variance**: Variance analysis with error bars

## Usage

### Enable Cross-Category Analysis

Update `configs/train_config.json`:
```json
{
  "data_config": {
    "use_multiple_categories": true,
    "category_balance": true,
    "cross_category_evaluation": true,
    "amazon_categories": [
      "Appliances",
      "Digital_Music", 
      "Gift_Cards",
      "Health_and_Personal_Care"
    ]
  }
}
```

### Run Cross-Category Analysis

#### Option 1: Full Workflow (Recommended)
```bash
# Run complete workflow including cross-category analysis
python run_multimodal_workflow.py

# Or run only evaluation (includes cross-category analysis)
python run_multimodal_workflow.py --steps eval
```

#### Option 2: From Parent Directory
```bash
# From anywhere
python run_multimodal_workflow.py
```

## Output Files

### Results Files
- `data/processed/cross_category_evaluation_results.json`: Detailed evaluation results
- `data/processed/cross_category_summary.json`: Summary report with key findings

### Visualizations
- `output/plots/category_performance_comparison.png`: Model performance across categories
- `output/plots/baseline_performance_comparison.png`: Baseline comparison across categories
- `output/plots/performance_variance.png`: Performance variance analysis

## Expected Results Structure

```json
{
  "category_performance": {
    "Appliances": {
      "HR@10": 0.342,
      "MRR": 0.187,
      "NDCG@10": 0.298,
      "Precision@10": 0.156
    },
    "Digital_Music": {
      "HR@10": 0.298,
      "MRR": 0.162,
      "NDCG@10": 0.267,
      "Precision@10": 0.142
    }
    // ... other categories
  },
  "baseline_performance": {
    "Appliances": {
      "Collaborative_Filtering": {"HR@10": 0.198, "MRR": 0.112},
      "Content_Based": {"HR@10": 0.156, "MRR": 0.087}
      // ... other baselines
    }
    // ... other categories
  },
  "comparison_analysis": {
    "best_performing_category": {
      "HR@10": {"category": "Appliances", "value": 0.342},
      "MRR": {"category": "Appliances", "value": 0.187}
    },
    "performance_variance": {
      "HR@10": {"mean": 0.320, "std": 0.022, "min": 0.298, "max": 0.342}
    }
  }
}
```

## Key Insights Generated

### 1. Performance Rankings
- Which category the model performs best on
- Which category the model struggles with
- Performance consistency across categories

### 2. Baseline Analysis
- Best baseline for each category
- Category-specific baseline performance patterns
- Model vs baseline comparison per category

### 3. Variance Analysis
- Performance consistency (low variance = consistent)
- Categories with high performance variance
- Recommendations for category-specific tuning

### 4. Recommendations
- Categories requiring model improvement
- Suggestions for category-specific approaches
- Insights for model generalization

## Example Analysis Output

```
CROSS-CATEGORY PERFORMANCE ANALYSIS SUMMARY
============================================================

📊 MODEL PERFORMANCE BY CATEGORY:
----------------------------------------

Appliances:
  HR@10: 0.3420
  MRR: 0.1870
  NDCG@10: 0.2980
  Precision@10: 0.1560

Digital_Music:
  HR@10: 0.2980
  MRR: 0.1620
  NDCG@10: 0.2670
  Precision@10: 0.1420

🏆 BEST PERFORMING CATEGORIES:
----------------------------------------
  HR@10: Appliances (0.3420)
  MRR: Appliances (0.1870)
  NDCG@10: Appliances (0.2980)
  Precision@10: Appliances (0.1560)

📊 PERFORMANCE VARIANCE:
----------------------------------------
  HR@10: 0.3200 ± 0.0220
    Range: 0.2980 - 0.3420
  MRR: 0.1745 ± 0.0125
    Range: 0.1620 - 0.1870
```

## Integration with Paper

This cross-category analysis directly addresses reviewer concerns about:
- **Single Domain Limitation**: Demonstrates performance across multiple categories
- **Generalizability**: Shows model performance consistency across domains
- **Comprehensive Evaluation**: Provides detailed category-specific insights

### Paper Sections to Update

1. **Results Section**: Add cross-category performance table
2. **Analysis Section**: Include category-specific insights
3. **Discussion**: Address generalizability across domains
4. **Limitations**: Discuss category-specific challenges

## Troubleshooting

### Common Issues

1. **No Category Data**: Ensure multi-category data is loaded
2. **Missing Predictions**: Run model evaluation first
3. **Configuration Issues**: Check `use_multiple_categories` setting

### Debug Mode
```bash
python run_cross_category_analysis.py --verbose
```

This cross-category analysis system provides comprehensive insights into model performance across different product categories, directly supporting your research goals of comparing model performance across various product categories.
