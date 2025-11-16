# Copy-Paste Augmentation CLI

This document describes all input options for `scripts/copy_paste_augment.py`.

## Synopsis

```
python scripts/copy_paste_augment.py \
  --targets Bombus_sandersoni Bombus_bohemicus Bombus_ternarius \
  [--per-class-add 300] \
  [--bg-mode inpaint|raw|solid] \
  [--solid-color #DCDCDC|auto] \
  [--solid-sample-count 200] \
  [--canvas-size 640] \
  [--fg-resize-policy downscale_to_fit|keep|ratio_range] \
  [--paste-position center|random] \
  [--sam-checkpoint checkpoints/sam_vit_h.pth] \
  [--dataset-root GBIF_MA_BUMBLEBEES]
```

## Options

- `--targets <str...>` (required)
  - One or more species folder names (space-separated), e.g.: `Bombus_sandersoni Bombus_bohemicus Bombus_ternarius`.
  - Augmentation is applied to `train/<species>/` only.

- `--per-class-add <int>` (default: `300`)
  - Number of augmented images to add per target species.

- `--bg-mode <inpaint|raw|solid>` (default: `inpaint`)
  - Background generation mode for compositing:
    - `inpaint`: pick a real train image, remove its central bee (SAM+inpaint), paste cutout.
    - `raw`: pick a real train image as-is, paste cutout directly.
    - `solid`: use a single-color canvas (see `--solid-color`, `--canvas-size`).

- `--solid-color <hex|auto>` (default: `#DCDCDC`)
  - Solid background color when `--bg-mode solid`.
  - Hex example: `#DCDCDC`.
  - Special values: `auto` (aliases: `average`, `auto_mean`) → sample solid colors from real backgrounds by computing mean RGB per background image (bee removed first).

- `--solid-sample-count <int>` (default: `200`)
  - Number of real background images to sample when `--solid-color auto` is used.
  - The script saves sampled colors to `RESULTS/copy_paste/bg_color_samples.json`.

- `--canvas-size <int>` (default: `640`)
  - Square canvas size for `--bg-mode solid`.

- `--fg-resize-policy <downscale_to_fit|keep|ratio_range>` (default: `downscale_to_fit`)
  - Foreground (bee cutout) sizing policy:
    - `downscale_to_fit`: keep original size unless it would overflow; then downscale to fit within ~90% of the canvas.
    - `keep`: never resize, even if it overflows (may clip at edges).
    - `ratio_range`: legacy behavior; randomly scale to 15–35% of background short side.

- `--paste-position <center|random>` (default: `center`)
  - Where to paste the foreground on the background.

- `--sam-checkpoint <path>` (default: `checkpoints/sam_vit_h.pth`)
  - Path to a Segment Anything (SAM) checkpoint.
  - Required for segmentation and for inpainting-bee removal in `inpaint` mode.

- `--dataset-root <path>` (default: `GBIF_MA_BUMBLEBEES`)
  - Root directory containing `prepared/` or `prepared_split/`.
  - The script scaffolds `prepared_cnp/` from these (if missing) and writes augmented images under `prepared_cnp/train/<species>/`.

## Behavior summary

- Cutouts (RGBA) are saved and reused:
  - `CACHE_CNP/cutouts/<species>/cutout_*.png`
- Augmented composites are saved to:
  - `GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/aug_*.png`
- Logs:
  - `RESULTS/copy_paste/generation_log.json`
  - (If `--solid-color auto`) `RESULTS/copy_paste/bg_color_samples.json`
- Background and source pools exclude previously augmented files: files prefixed with `aug_` are ignored.
- The compositor avoids pasting onto the exact same original image that produced the cutout.

## Examples

- Solid backgrounds sampled from real backgrounds (auto mean color), center paste, keep original foreground size unless too big:
```
python scripts/copy_paste_augment.py \
  --targets Bombus_sandersoni Bombus_bohemicus Bombus_ternarius \
  --per-class-add 300 \
  --bg-mode solid \
  --solid-color auto \
  --solid-sample-count 200 \
  --canvas-size 640 \
  --fg-resize-policy downscale_to_fit \
  --paste-position center \
  --sam-checkpoint checkpoints/sam_vit_h.pth
```

- Inpainted real backgrounds, random placement, legacy random foreground scaling:
```
python scripts/copy_paste_augment.py \
  --targets Bombus_sandersoni \
  --per-class-add 150 \
  --bg-mode inpaint \
  --fg-resize-policy ratio_range \
  --paste-position random \
  --sam-checkpoint checkpoints/sam_vit_h.pth
```
