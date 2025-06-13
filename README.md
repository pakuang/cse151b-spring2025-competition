# Climate Emulator – ConvLSTM-UNet Hybrid

This repository contains code for a ConvLSTM-UNet hybrid architecture developed for the [CSE 151B Spring 2025](https://sites.google.com/view/cse151b-251b/151b-info) climate emulation competition. It builds on the official [starter code](https://www.kaggle.com/t/6f53c429d53099dc7cc590f9bf390b10) and extends it with temporal modeling, attention-based encoders, and baseline ablations.

---

## Kaggle Competition

**[CSE 151B Climate Emulation Challenge](https://www.kaggle.com/t/6f53c429d53099dc7cc590f9bf390b10)**  
Goal: Emulate a climate simulation model to predict monthly precipitation (`pr`) and surface air temperature (`tas`) under different emissions scenarios.

---

## Project Overview

We implemented a progression of increasingly expressive models, starting from a basic CNN and culminating in a ConvLSTM-enhanced UNet. Key contributions include:

- Temporal windowing (stacked inputs)
- ConvLSTM bottleneck for short-term memory
- Dual decoder heads for disentangled prediction
- Comparison of spatial attention variants (SE, CBAM, ViT)

---

## Directory Structure

- `main.py`: Entry point for single-frame (non-temporal) models.
- `main-temporal.py`: Entry point for temporal models using input sequences.
- `src/models/`: Model definitions including ConvLSTM-UNet and variants.
- `configs/`: Hydra configs for model/data/training setup.
- `notebooks/`: Visualization and exploratory notebooks.

---

## Dataset

The dataset is derived from CMIP6 outputs and downsampled to a `(48 × 72)` global grid.  

### Inputs (forcings):
- `CO2`, `CH4`, `SO2`, `BC`, `rsdt`

### Outputs (targets):
- `tas` (surface air temperature, K)  
- `pr` (precipitation rate, mm/day)

### Scenario Splits:
- **Train**: SSP126, SSP370 (early), SSP585  
- **Val**: Last 10 years of SSP370  
- **Test**: Last 10 years of SSP245  

Only ensemble member 0 is used for consistency.

---

## | Getting Started

1. Create a fresh virtual environment (we recommend python >= 3.10) and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the zarr data files from the competition page and place them in a directory of your choice (you'll later need to specify it with the ``data.path`` argument).

### Configuration

This project uses Hydra for configuration management. The main configuration files are in the `configs/` directory:

- `configs/main_config.yaml`: Main configuration file that includes other configuration files
- `configs/data/default.yaml`: Dataset and data-loading related settings (e.g. data path and batch size)
- `configs/model/simple_cnn.yaml`: Model architecture settings (e.g. architecture type, number of layers)
- `configs/training/default.yaml`: Training parameters (e.g. learning rate)
- `configs/trainer/default.yaml`: PyTorch Lightning Trainer settings (e.g. number of GPUs, precision)

### Running the Model

This codebase uses PyTorch Lightning for training. It is meant to be a starting point for your own model development.
You may use any (or none) of the code provided here and are free to modify it as needed.

To train the model with default settings:

```bash
python main.py data.path=/path/to/your/data.zarr
```

#### Logging

It is recommended to use Weights & Biases for logging.
To enable logging, set `use_wandb=true` and specify your W&B (team) username with `wandb_entity=<your-wandb-username>`.
You will need to create a project `cse-151b-competition` on Weights & Biases. 
When logging is enabled, the training script will automatically log metrics, and hyperparameters to your W&B project.
This will allow you to monitor your training runs and compare different experiments more conveniently from the W&B dashboard.

#### Common Configuration Options

Override configuration options from the command line:

```bash
# Use Weights & Biases for logging (recommended). Be sure to first create a project ``cse-151b-competition`` on wandb.
python main.py data.path=/path/to/your/data.zarr use_wandb=true wandb_entity=<your-wandb-username>

# Change batch size and learning rate and use different batch size for validation
python main.py data.path=/path/to/your/data.zarr data.batch_size=64 data.eval_batch_size=32 training.lr=1e-3

# Change the number of epochs
python main.py data.path=/path/to/your/data.zarr trainer.max_epochs=200

# Train on 4 GPUs with DistributedDataParallel (DDP) mode
python main.py data.path=/path/to/your/data.zarr trainer.strategy="ddp_find_unused_parameters_false" trainer.devices=4 

# Resume training from (or evaluate) a specific checkpoint
python main.py data.path=/path/to/your/data.zarr ckpt_path=/path/to/your/checkpoint.ckpt
```

## Acknowledgments

Portions of this repository and README.md were adapted from the official CSE151B Spring 2025 starter codebase. The following additions were made:

- `main-temporal.py`: Custom temporal dataloader for multi-month input windows  
- ConvLSTM-UNet hybrid architecture for spatiotemporal modeling  
- Integration of spatial attention modules (CBAM, SE, ViT)  
- Customized evaluation pipeline for validation metrics and leaderboard reporting  
- `notebooks/data-exploration.ipynb`: additional data exploration built upon starter
