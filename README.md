# Enhancing Product Recommendation with Multimodal LLMs

This repository contains the implementation for the paper **"Enhancing Product Recommendation with Multimodal LLMs"**, which explores the use of multimodal language models to improve product recommendation systems by incorporating both textual and visual information.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Configuration](#configuration)
- [Vision Models](#vision-models)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Overview

Traditional recommendation systems often rely solely on user-item interaction data or textual descriptions. This work introduces a novel approach that leverages multimodal language models to process both product images and textual information, leading to more accurate and contextually aware product recommendations.

### Research Contributions

- **Multimodal Fusion**: Integration of visual and textual features for enhanced recommendation accuracy
- **Vision Model Flexibility**: Support for multiple vision encoders (ResNet, MobileNet, EfficientNet)
- **Efficient Training**: Implementation of LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **Comprehensive Evaluation**: Multi-metric assessment of recommendation quality

## Key Features

- **Multimodal Processing**: Combines product images and textual descriptions
- **Flexible Vision Models**: Support for ResNet, MobileNet, and EfficientNet architectures
- **LoRA Integration**: Parameter-efficient fine-tuning for large language models
- **Comprehensive Metrics**: Multiple evaluation metrics for recommendation quality
- **Configurable Pipeline**: Easy-to-modify configuration for different experiments
- **Visualization Tools**: Built-in plotting and analysis utilities
- **Safe Data Management**: Automatic backup creation when limiting dataset size
- **Utility Scripts**: Easy-to-use tools for dataset and model management

## Architecture

The system consists of several key components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Text Input    │    │  Image Input    │    │   User History  │
│   (Reviews,     │    │  (Product       │    │   (Purchase     │
│   Descriptions) │    │   Images)       │    │   History)      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Text Encoder   │    │ Vision Encoder  │    │  Context        │
│  (GPT-2)        │    │ (ResNet/Mobile) │    │  Encoder        │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────┬───────────┴──────────┬───────────┘
                     ▼                      ▼
            ┌─────────────────────────────────────────┐
            │         Multimodal Fusion Layer         │
            └─────────────────┬───────────────────────┘
                              ▼
            ┌─────────────────────────────────────────┐
            │         Recommendation Head             │
            │      (Next Product Prediction)          │
            └─────────────────┬───────────────────────┘
                              ▼
            ┌─────────────────────────────────────────┐
            │           Output: Product IDs           │
            └─────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/multimodal_llm_recsys_amazon.git
   cd multimodal_llm_recsys_amazon
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

## Data Preparation

### Amazon Product Data

The system is designed to work with Amazon product review and metadata data. The data processing pipeline handles:

- **Review Data**: User reviews, ratings, and timestamps
- **Product Metadata**: Titles, descriptions, categories, and image URLs
- **Image Processing**: Automatic download and preprocessing of product images

### Data Structure

```
data/
├── raw/                    # Raw data files
│   ├── reviews.jsonl       # Amazon review data
│   └── metadata.jsonl      # Product metadata
├── processed/              # Processed data
│   ├── train.json          # Training split
│   ├── val.json            # Validation split
│   ├── test.json           # Test split
│   ├── *.json.backup       # Backup files (if using dataset limiting)
│   └── images/             # Downloaded product images
└── image_cache/            # Cached image embeddings
```

### Processing Commands

```bash
# Process raw data
python -m src.data.prepare_data

# This will:
# 1. Load and clean review data
# 2. Download product images
# 3. Create train/val/test splits
# 4. Generate multimodal datasets
```

### Dataset Size Limiting

For large datasets, you can limit the number of entries to reduce processing time and memory usage. The system provides safe options that preserve your original data:

#### Option 1: Create Backups (Recommended)

```bash
# Limit to 40,000 entries with automatic backups
python limit_processed_data.py 40000
```

This approach:
- Creates backup files (`.json.backup`) of your original data
- Limits the main files to the specified number of entries
- Uses random sampling for representative subsets
- Preserves your original 100k+ data safely

**File structure after limiting:**
```
data/processed/
├── train.json          # 40k entries (limited)
├── train.json.backup   # 100k entries (original preserved)
├── val.json            # 40k entries (limited)
├── val.json.backup     # 100k entries (original preserved)
├── test.json           # 40k entries (limited)
└── test.json.backup    # 100k entries (original preserved)
```

#### Option 2: Create New Files in Different Directory

```bash
# Preserve all original files and create limited versions elsewhere
python limit_processed_data.py 40000 --output-dir data/processed_40k
```

#### Restoring Original Data

If you need to restore your original dataset:

```bash
# Restore original 100k+ data
mv data/processed/train.json.backup data/processed/train.json
mv data/processed/val.json.backup data/processed/val.json
mv data/processed/test.json.backup data/processed/test.json
```

#### Configuration-Based Limiting

You can also set limits in the configuration file:

```json
{
    "model_config": {
        "test_mode": true,
        "test_size": 40000
    },
    "data_config": {
        "max_entries": 40000
    }
}
```

Then re-run data preparation:
```bash
python -m src.data.prepare_data
```

## Training

### Quick Start

```bash
# Train with default configuration (ResNet-18 vision model)
python -m src.training.train

# Train with different vision model
python -m src.training.train --config configs/train_config.json
```

### Utility Scripts

The repository includes several utility scripts for data management:

#### Dataset Size Limiting

```bash
# Limit processed data to 40,000 entries (creates backups)
python limit_processed_data.py 40000

# Limit with custom output directory
python limit_processed_data.py 40000 --output-dir data/processed_40k

# Limit specific file only
python limit_processed_data.py 40000 --file data/processed/train.json

# Overwrite without backup (not recommended)
python limit_processed_data.py 40000 --no-backup
```

#### Training Checkpointing and Resumption

The system automatically saves checkpoints during training to prevent data loss:

```bash
# Checkpoints are automatically saved to output/model/training_checkpoint.pth
# Resume training from the latest checkpoint
python resume_training.py

# List available checkpoints
python resume_training.py --list

# Resume from specific checkpoint
python resume_training.py output/model/training_checkpoint.pth
```

**What's saved in checkpoints:**
- Model weights and state
- Optimizer state
- Learning rate scheduler state
- Training/validation losses
- Current epoch and step
- All training progress

#### Vision Model Selection

```bash
# Interactive model selection
python limit_dataset.py

# Set specific model
python limit_dataset.py 40000  # Sets dataset size to 40k
```

### Configuration

The training can be customized through `configs/train_config.json`:

```json
{
    "model_config": {
        "model_name": "gpt2-medium (355M)",
        "context_length": 1024,
        "vocab_size": 50257
    },
    "custom_config": {
        "phi3": true,
        "vision_model": "resnet-18",
        "lora": true
    },
    "training_config": {
        "num_epochs": 5,
        "learning_rate": 5e-6,
        "batch_size": 4
    }
}
```

### Vision Models

The system supports multiple vision encoders:

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| `resnet-18` | ~44MB | Fast | Development |
| `resnet-50` | ~98MB | Medium | Production |
| `mobilenet-v2` | ~14MB | Very Fast | Mobile |
| `mobilenet-v3` | ~7MB | Very Fast | Edge |
| `efficientnet-b0` | ~29MB | Fast | Balanced |

See [vision_models.md](vision_models.md) for detailed information.

## Evaluation

### Two-Track Evaluation System

The system uses a sophisticated two-track evaluation approach that separates two different skills:

#### Track A: Item Recommendation Quality (Exclude EOS)
Measures the model's ability to recommend actual products, excluding `<|endoftext|>` tokens:

- **Item Precision@K**: Accuracy of top-K product recommendations
- **Item Recall@K**: Coverage of relevant products in top-K
- **Item NDCG@K**: Normalized Discounted Cumulative Gain for products
- **Item MRR**: Mean Reciprocal Rank for products
- **Item Coverage@K**: Percentage of catalog items recommended
- **Item Diversity@K**: Recommendation diversity measures
- **Item Novelty@K**: Novelty of recommended products

#### Track B: Stop Prediction Quality (EOS Prediction)
Measures the model's ability to correctly predict when to stop recommending:

- **Stop Hit@K**: Whether model correctly predicts EOS when ground truth has EOS
- **Stop NDCG@K**: Position-sensitive metric for EOS prediction accuracy
- **Stop Rank**: Rank of first EOS prediction
- **Stop Prediction Rate**: Percentage of predictions that include EOS

### Enhanced Evaluation Pipeline

The system includes a comprehensive evaluation pipeline with multiple components:

```bash
# Run complete evaluation pipeline
python run_enhanced_workflow.py

# Run specific evaluation steps
python run_enhanced_workflow.py --steps eval
python run_enhanced_workflow.py --steps baseline
python run_enhanced_workflow.py --steps cross
```

### Evaluation Components

1. **Robust Evaluation**: Multiple runs with statistical testing and confidence intervals
2. **Baseline Comparison**: Traditional and multimodal baseline methods
3. **Ablation Studies**: Text-only, image-only, and fusion method analysis
4. **Cross-Category Analysis**: Performance across different product categories
5. **Main Evaluation**: Comprehensive results consolidation

### Baselines

The system includes comprehensive baseline methods for comparison:

#### Traditional Baselines
- **Collaborative Filtering**: Matrix factorization approach
- **Content-Based**: TF-IDF similarity recommendations
- **Popularity-Based**: Most popular items recommendations
- **Matrix Factorization**: SVD-based recommendations

#### Multimodal Baselines
- **VBPR**: Visual Bayesian Personalized Ranking
- **DeepCoNN**: Deep Cooperative Neural Networks
- **NRMF**: Neural Rating Matrix Factorization
- **SASRec**: Self-Attentive Sequential Recommendation

## Results

### Two-Track Evaluation Results

The evaluation reveals the model's performance across two distinct skills:

#### Track A: Item Recommendation Quality
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Item Precision@10 | 0.00018 (0.018%) | Very low precision - few relevant recommendations |
| Item Recall@10 | 0.00057 (0.057%) | Low recall - missing many relevant items |
| Item Hit Rate@10 | 0.0018 (0.18%) | Low hit rate - rarely finding relevant items |
| Item NDCG@10 | 0.00051 (0.051%) | Poor ranking quality |
| Item MRR | 0.00094 (0.094%) | Low reciprocal rank |
| Item Coverage@10 | 0.117 (11.7%) | Good catalog coverage |
| Item Diversity@10 | 0.920 (92%) | Excellent recommendation diversity |

#### Track B: Stop Prediction Quality
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Stop Hit@10 | 0.0356 (3.56%) | Low stop prediction accuracy |
| Stop NDCG@10 | 0.0688 (6.88%) | Poor position-sensitive stopping |
| Stop Prediction Rate | 0.0386 (3.86%) | Rarely predicts stops |
| Stop Rank (median) | 1.0 | When stops are predicted, they're at rank 1 |

### Key Findings

1. **Conservative Model Behavior**: The model is extremely conservative, rarely recommending actual products
2. **Poor Item Recommendation**: Very low precision and recall for actual product recommendations
3. **Excellent Diversity**: When the model does recommend products, they are highly diverse (92%)
4. **Good Coverage**: The model covers 11.7% of the available catalog
5. **Low Stop Prediction**: The model rarely predicts when to stop recommending

### Model Performance Analysis

The two-track evaluation reveals that the current model:
- **Prioritizes safety over recommendations** (very conservative approach)
- **Struggles with actual product recommendations** (low hit rate)
- **Provides diverse recommendations** when it does recommend
- **Has room for significant improvement** in both tracks

### Recommendations for Improvement

1. **Increase recommendation confidence**: Model is too conservative
2. **Balance stopping vs. recommending**: Current model rarely recommends
3. **Fine-tune on item recommendation**: Focus training on actual products
4. **Implement better data filtering**: Improve training data quality
5. **Consider ensemble methods**: Combine multiple approaches

## Configuration

### Model Configuration

```json
{
    "model_config": {
        "model_name": "gpt2-medium (355M)",
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.01,
        "qkv_bias": true
    }
}
```

### Training Configuration

```json
{
    "training_config": {
        "num_epochs": 5,
        "learning_rate": 5e-6,
        "weight_decay": 0.1,
        "batch_size": 4,
        "lora": true,
        "lora_rank": 8
    }
}
```

### Data Configuration

```json
{
    "data_config": {
        "batch_size": 4,
        "num_workers": 0,
        "number_of_items_to_predict": 10,
        "min_interactions": 3
    }
}
```

python -m src.evaluation.metrics 
python -m src.evaluation.process_model_outputs
python -m src.evaluation.compare_baselines 


## Vision Models

The system supports multiple vision encoders for different use cases:

- **Development**: Use `resnet-18` for fast iteration
- **Production**: Use `resnet-50` for better accuracy
- **Mobile/Edge**: Use `mobilenet-v3` for resource constraints.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{multimodal_recsys_2024,
  title={Enhancing Product Recommendation with Multimodal LLMs},
  author={Babaniyi Olaniyi},
  journal={arXiv preprint},
  year={2025}
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Amazon for providing the product review dataset
- Hugging Face for the transformer models and vision encoders
- The open-source community for various tools and libraries

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact the authors at [your-email@domain.com]

---


# 1. Backup current results
cp data/processed/result_metrics.json data/processed/result_metrics_backup.json

# 2. Run complete evaluation pipeline
python run_enhanced_workflow.py --steps process eval baseline cross

# 3. Compare results
python compare_results.py

# 4. Detailed analysis  
python analyze_improvements.py

**Note**: This repository is part of ongoing research. Results may vary based on data preprocessing and hardware configuration. 