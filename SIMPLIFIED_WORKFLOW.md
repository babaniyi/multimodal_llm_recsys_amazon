# Simplified Workflow Guide

The workflow has been simplified to use a single script that runs everything automatically, including cross-category analysis.

## Quick Start

### Run Everything (Recommended)
```bash
# From the project directory
python run_multimodal_workflow.py

# Or from anywhere
python run_multimodal_workflow.py
```

### Run Specific Steps
```bash
# Data preparation only
python run_multimodal_workflow.py --steps data

# Training only
python run_multimodal_workflow.py --steps train

# Evaluation only (includes cross-category analysis)
python run_multimodal_workflow.py --steps eval

# Multiple steps
python run_multimodal_workflow.py --steps data train eval
```

## What Gets Run Automatically

When you run the full workflow, it automatically executes:

1. **Data Preparation** (`--steps data`)
   - Downloads multi-category data (Appliances, Digital_Music, Gift_Cards, Health_and_Personal_Care)
   - Processes and splits the data
   - Creates train/val/test splits

2. **Training** (`--steps train`)
   - Trains the multimodal LLM model
   - Uses LoRA for efficient fine-tuning
   - Saves the trained model

3. **Model Processing** (`--steps process`)
   - Processes model outputs
   - Runs ablation studies (if enabled)
   - Generates predictions

4. **Evaluation** (`--steps eval`)
   - Compares with baselines
   - Runs robust evaluation with statistical testing
   - **Cross-category analysis** (automatically included)
   - Creates performance visualizations

## Cross-Category Analysis

The cross-category analysis now runs automatically as part of the evaluation step. It:

- ✅ Evaluates model performance across all 4 categories
- ✅ Compares baselines across categories
- ✅ Identifies best/worst performing categories
- ✅ Creates performance visualizations
- ✅ Generates comprehensive reports

## Output Files

After running the workflow, you'll find:

### Results
- `data/processed/result_metrics.json` - Main evaluation results
- `data/processed/cross_category_evaluation_results.json` - Cross-category analysis
- `data/processed/cross_category_summary.json` - Cross-category summary
- `data/processed/robust_evaluation_results.json` - Statistical evaluation
- `data/processed/ablation_study_results.json` - Ablation studies

### Visualizations
- `output/plots/category_performance_comparison.png` - Category performance
- `output/plots/baseline_performance_comparison.png` - Baseline comparison
- `output/plots/performance_variance.png` - Performance variance

## Configuration

The workflow uses `configs/train_config.json`. Key settings:

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
  },
  "training_config": {
    "statistical_testing": true,
    "evaluation_runs": 5,
    "ablation_studies": true
  }
}
```

## Command Line Options

```bash
python run_multimodal_workflow.py [OPTIONS]

Options:
  --skip-training          Skip training step (if model exists)
  --steps STEPS           Run specific steps: data, train, process, eval
  --verbose, -v           Enable verbose logging
  --force-training        Force retraining even if model exists
  --config CONFIG         Use custom config file
```

## Example Usage

```bash
# Full workflow with verbose output
python run_multimodal_workflow.py --verbose

# Skip training, run evaluation only
python run_multimodal_workflow.py --skip-training --steps eval

# Force retraining
python run_multimodal_workflow.py --force-training

# Use custom config
python run_multimodal_workflow.py --config my_config.json
```

## Troubleshooting

### Common Issues

1. **Missing Data**: Run with `--steps data` first
2. **Missing Model**: Run with `--steps train` first
3. **Permission Errors**: Ensure write permissions for output directories

### Debug Mode
```bash
python run_multimodal_workflow.py --verbose
```

## Integration with Paper

The simplified workflow generates all the results needed for your paper:

- ✅ Cross-category performance comparison
- ✅ Statistical significance testing
- ✅ Ablation study results
- ✅ Baseline comparisons
- ✅ Performance visualizations

All results are automatically saved and ready for inclusion in your paper sections.

## Summary

The workflow is now simplified to a single command that runs everything:

```bash
python run_multimodal_workflow.py
```

This single command will:
1. Download and process multi-category data
2. Train the multimodal model
3. Run comprehensive evaluation
4. Perform cross-category analysis
5. Generate all visualizations and reports

Perfect for getting complete results with minimal effort!


