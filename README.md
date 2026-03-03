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

For copy-paste augmentation (SAM-based), install segment-anything and download the checkpoint:
```bash
pip install segment-anything
mkdir -p checkpoints
wget -O checkpoints/sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

## Project Structure

```
bumblebee_bplusplus/
├── run.py                   # Thin orchestrator — dispatches to pipeline modules
├── configs/
│   ├── training_config.yaml # Hyperparameters (epochs, batch_size, backbone, …)
│   └── species_config.json  # MA bumblebee species list
├── scripts/
│   ├── assemble_dataset.py  # Assemble baseline + augmented images into training datasets
│   └── llm_judge.py         # Two-stage LLM-as-judge evaluation for synthetic images
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
| `python run.py evaluate --type bioclip` | BioCLIP PCA/t-SNE plots |
| `python run.py evaluate --type mllm` | Multimodal LLM zero-shot classification |
| `python run.py all` | Full pipeline end-to-end |

### Dataset assembly

After generating synthetic or copy-paste images, use `assemble_dataset.py` to build
a training dataset that combines the baseline with augmented images. Synthetic images
are automatically resized to match YOLO-crop dimensions (short edge = 640).

```bash
# D3: unfiltered synthetic — randomly select from all generated images
python scripts/assemble_dataset.py --mode unfiltered --target 300 --name d3_synthetic

# D5: LLM-judge filtered — only use images that passed quality evaluation
python scripts/assemble_dataset.py --mode llm_filtered --target 300 \
    --judge-results RESULTS/llm_judge_eval/results.json --name d5_llm_filtered

# D4: copy-paste augmentation (point --synthetic-dir at copy-paste output)
python scripts/assemble_dataset.py --mode unfiltered --target 300 \
    --synthetic-dir RESULTS/cnp_generation/train --name d4_cnp

# Re-assemble with --force to overwrite an existing dataset
python scripts/assemble_dataset.py --mode unfiltered --target 300 --name d3_synthetic --force
```

### LLM-judge evaluation

```bash
# Evaluate synthetic image quality with two-stage GPT-4o rubric
python scripts/llm_judge.py --image-dir RESULTS/synthetic_generation
python scripts/llm_judge.py --species Bombus_ashtoni --output-dir RESULTS/llm_judge_eval
```

### Or run modules directly (supports `--help`)

```bash
python pipeline/train/simple.py --dataset cnp_100 --backbone resnet101 --epochs 50
python pipeline/train/hierarchical.py --dataset prepared_split
python pipeline/augment/synthetic.py --species Bombus_ashtoni --count 30
python pipeline/evaluate/metrics.py --models baseline d3_synthetic d4_cnp
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
├── prepared_split/          # Baseline dataset
│   ├── train/               # 70%
│   ├── valid/               # 15%
│   └── test/                # 15%
├── prepared_d3_synthetic/   # Baseline + unfiltered synthetic (assembled)
├── prepared_d4_cnp/         # Baseline + copy-paste (assembled)
└── prepared_d5_llm_filtered/# Baseline + LLM-filtered synthetic (assembled)

RESULTS/
├── synthetic_generation/    # Raw GPT-image-1 output (1024x1024)
├── cnp_generation/          # Raw copy-paste composites
├── llm_judge_eval/          # LLM judge results + visualizations
└── <run_name>_gbif/
    ├── best_multitask.pt    # Best checkpoint (hierarchical)
    ├── best_multitask_focus.pt  # Best focus-species checkpoint (C1b)
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
