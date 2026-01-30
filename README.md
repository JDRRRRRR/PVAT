<<<<<<< HEAD
# PVAT: Patch- and Variable-Aligned Transformer for Structurally Heterogeneous Federated Time-Series Forecasting

> **Submitted to IJCNN 2026 (Under Review)**
>
> **Authors:** Anonymous

## Abstract

Federated learning offers a promising paradigm for leveraging distributed time-series data without compromising privacy. However, a fundamental obstacle arises when nodes employ different sampling devices: heterogeneous sampling rates produce time series of varying granularities, and diverse sensor configurations yield inconsistent variable sets across nodes. These structural heterogeneities prevent direct parameter aggregation in conventional federated frameworks.

We propose **PVAT (Patch- and Variable-Aligned Transformer)**, a Transformer-based forecasting model that reconciles structurally heterogeneous data with homogeneous federated aggregation. PVAT introduces two alignment mechanisms:

1. **Patch Embedding for Temporal Alignment**: Segments raw series by fixed physical time intervals and projects variable-length patches into uniform-dimensional tokens, enabling temporal alignment across different sampling rates.

2. **Variable Embedding Table for Semantic Alignment**: A globally synchronized table that assigns learnable semantic vectors to each variable category, ensuring consistent variable semantics network-wide.

<p align="center">
  <img src="fig/PVAT_1_01.png" width="80%">
</p>

**Figure 1.** Structural heterogeneity in federated time-series forecasting. Nodes with diverse sampling rates and variable sets must collaboratively train a shared model.

<p align="center">
  <img src="fig/PVAT_2_01.png" width="95%">
</p>

**Figure 2.** The architecture of the proposed PVAT model. PVAT reconciles structural heterogeneity across nodes through Patch-wise Temporal Alignment and Global Variable Alignment mechanisms.

## Requirements

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.8.0
```

## Project Structure

```
PVAT-code/
├── dataset/           # Data loading and preprocessing
├── models/            # Model implementations (PVAT, baselines)
├── layers/            # Neural network layers
├── utils/             # Utility functions and metrics
├── scripts/           # Training scripts
├── evaluation/        # Evaluation results
├── fig/               # Figures
└── run.py             # Main training script
```

## Datasets

We use eight widely-adopted benchmarks:
- **ETT** (ETTh1, ETTh2, ETTm1, ETTm2): Electricity Transformer Temperature
- **Electricity**: Electricity consumption
- **Traffic**: Road occupancy rates
- **Weather**: Weather observations
- **Exchange**: Exchange rates

Download datasets and place them in the `dataset/` directory.

## Usage

### Basic Training

```bash
python run.py --model PVAT --data ETTh1 --seq_len 96 --pred_len 96
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | PVAT | Model architecture: PVAT, iTransformer, PatchTST, TimeXer, TimesNet, TimeMixer, DLinear |
| `--data` | ETTh1 | Dataset: ETTh1, ETTh2, ETTm1, ETTm2, electricity, traffic, weather, exchange_rate |
| `--root_path` | dataset/ETT-small/ | Root directory containing data files |
| `--seq_len` | 96 | Input sequence length (lookback window) |
| `--pred_len` | 96 | Prediction horizon length |
| `--patch_len` | 16 | Patch size for patch-based models |
| `--features` | M | Forecasting task: M (multivariate), S (univariate), MS (multivariate-to-single) |
| `--d_model` | 512 | Model embedding dimension |
| `--n_heads` | 8 | Number of attention heads |
| `--en_layers` | 2 | Number of encoder layers |
| `--de_layers` | 2 | Number of decoder layers |
| `--batch_size` | 32 | Training batch size |
| `--train_epochs` | 10 | Number of training epochs |
| `--learning_rate` | 0.0001 | Learning rate |
| `--gpu` | None | GPU device ID (auto-detects if not specified) |

### Examples

**Multivariate forecasting on ETTh1:**
```bash
python run.py --model PVAT --data ETTh1 --features M --seq_len 96 --pred_len 96
```

**Multivariate-to-univariate forecasting on Electricity:**
```bash
python run.py --model PVAT --data electricity --features MS --seq_len 96 --pred_len 192 --root_path dataset/electricity/
```

**Training with different prediction horizons:**
```bash
# Short-term forecasting
python run.py --model PVAT --data ETTm1 --pred_len 96

# Long-term forecasting
python run.py --model PVAT --data ETTm1 --pred_len 720
```

### Federated Learning Setup

For federated experiments with multiple GPUs:
```bash
torchrun --nproc_per_node=8 run.py --model PVAT --data ETTh1 --features MS
```

## Acknowledgments

We appreciate the following open-source repository for providing valuable code base and benchmarks:

- **Time-Series-Library**: https://github.com/thuml/Time-Series-Library

## Citation

```bibtex
@inproceedings{pvat2026,
  title={PVAT: Patch- and Variable-Aligned Transformer for Structurally Heterogeneous Federated Time-Series Forecasting},
  author={Anonymous},
  booktitle={International Joint Conference on Neural Networks (IJCNN)},
  year={2026}
}
```
=======
# PVAT
PVAT: Patch- and Variable-Aligned Transformer for Structurally Heterogeneous Federated Time-Series Forecasting
>>>>>>> d441922a821e9a6ad4e61a89f7a9c0f398386b44
