# Bumblebee Classification Pipeline

Research pipeline for fine-grained classification of Massachusetts bumblebee species using
hierarchical multi-task learning, copy-paste augmentation, and GPT-4o-based synthetic image generation.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9+. GPU (CUDA) recommended for training.

Copy your OpenAI key if using synthetic generation or LLM-judge evaluation:
```bash
export OPENAI_API_KEY="sk-..."
```

---

## Project Structure

```
bumblebee_bplusplus/
├── run.py                   # Thin orchestrator — dispatches to pipeline modules
├── configs/
│   ├── training_config.yaml # Hyperparameters (epochs, batch_size, backbone, …)
│   └── species_config.json  # MA bumblebee species list
└── pipeline/
    ├── config.py            # Shared config loader + path constants
    ├── collect.py           # Download GBIF images
    ├── analyze.py           # Dataset analysis + class-imbalance report
    ├── prepare.py           # YOLO detection, crop, 80/20 split
    ├── split.py             # Reorganise into 70/15/15 train/valid/test
    ├── augment/
    │   ├── copy_paste.py    # SAM-based cutout extraction + composite generation
    │   └── synthetic.py     # GPT-image-1 synthetic image generation
    ├── train/
    │   ├── simple.py        # Single-head ResNet classifier
    │   └── hierarchical.py  # 3-branch (Family/Genus/Species) ResNet50 via bplusplus
    └── evaluate/
        ├── metrics.py       # Auto-discover model checkpoints + comparison report
        ├── llm_judge.py     # GPT-4o structured rubric scoring of generated images
        └── bioclip.py       # BioCLIP embedding extraction + PCA/t-SNE visualisation
```

Every module is **both** a standalone CLI script and an importable `run()` API.

---

## Usage

### Via the orchestrator

| Command | What it does |
|---|---|
| `python run.py collect` | Download GBIF images for MA species |
| `python run.py analyze` | Report species counts and class imbalance |
| `python run.py prepare` | YOLO crop + 80/20 train/valid split |
| `python run.py split` | Reorganise into 70/15/15 train/valid/test |
| `python run.py augment --method copy_paste --count 100` | Copy-paste composites |
| `python run.py augment --method synthetic --count 50` | GPT-4o synthetic images |
| `python run.py train --type simple` | Train single-head ResNet |
| `python run.py train --type hierarchical` | Train hierarchical model |
| `python run.py evaluate --type metrics` | Run all model checkpoints, compare |
| `python run.py evaluate --type llm_judge` | LLM-as-judge image quality scoring |
| `python run.py evaluate --type bioclip` | BioCLIP PCA/t-SNE plots |
| `python run.py all` | Full pipeline end-to-end |

### Or run modules directly (supports `--help`)

```bash
python pipeline/train/simple.py --dataset cnp_100 --backbone resnet101 --epochs 50
python pipeline/train/hierarchical.py --dataset prepared_split
python pipeline/augment/synthetic.py --species Bombus_ashtoni --count 30
python pipeline/evaluate/metrics.py --models baseline cnp_100 synthetic_50
python pipeline/evaluate/llm_judge.py --dataset synthetic_100
python pipeline/evaluate/bioclip.py
```

---

## Data Layout

```
GBIF_MA_BUMBLEBEES/
├── <species>/               # Raw downloaded images (per species)
├── prepared/
│   ├── train/               # 80% (YOLO-cropped)
│   └── valid/               # 20%
└── prepared_split/
    ├── train/               # 70%
    ├── valid/               # 15%
    └── test/                # 15%

RESULTS/
└── <run_name>/
    ├── best_multitask.pt    # Best checkpoint (hierarchical)
    ├── model_best.pth       # Best checkpoint (simple)
    ├── training_log.json
    └── metrics.json
```

---

## Configuration

Edit `configs/training_config.yaml` to change hyperparameters — any `run()` call reads it
automatically. CLI flags override the YAML values.

---

## Extending the Pipeline

| Goal | Action |
|---|---|
| New augmentation method | Add `pipeline/augment/<name>.py` + wire `--method <name>` in `run.py` |
| New model architecture | Add `pipeline/train/<name>.py` + wire `--type <name>` in `run.py` |
| New evaluation | Add `pipeline/evaluate/<name>.py` + wire `--type <name>` in `run.py` |
