#!/usr/bin/env python3
"""
Visualize LLM judge results as an interactive HTML grid viewer and PNG contact sheets.

Produces:
  - viewer.html — self-contained HTML with base64-embedded thumbnail grid
  - contact_{species}.png — pass/fail contact sheets per species

Tier classification (from build_expert_validation.py):
  - strict_pass:  matches_target + diag=species + morph>=4.0
  - borderline:   matches_target + diag=species + 3.0<=morph<4.0
  - soft_fail:    matches_target + diag<species
  - hard_fail:    NOT matches_target

CLI
---
    python scripts/visualize_judge_results.py
    python scripts/visualize_judge_results.py --results RESULTS/llm_judge_eval/results.json --image-dir RESULTS/synthetic_generation
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import RESULTS_DIR

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

DEFAULT_RESULTS = RESULTS_DIR / "llm_judge_eval" / "results.json"
DEFAULT_IMAGE_DIR = RESULTS_DIR / "synthetic_generation"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "llm_judge_eval"

STRICT_MORPH_THRESHOLD = 4.0
BORDERLINE_MORPH_THRESHOLD = 3.0
THUMBNAIL_WIDTH = 200

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

TIER_COLORS = {
    "strict_pass": "#4CAF50",
    "borderline": "#FFEB3B",
    "soft_fail": "#FF9800",
    "hard_fail": "#F44336",
}

TIER_ORDER = ["strict_pass", "borderline", "soft_fail", "hard_fail"]


# ── Reused from build_expert_validation.py ────────────────────────────────────

def morph_mean(r: dict) -> float:
    morph = r.get("morphological_fidelity", {})
    scores = [
        v["score"] for v in morph.values()
        if isinstance(v, dict) and "score" in v and not v.get("not_visible", False)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def classify_tier(r: dict) -> str:
    """Classify a judge result into a quality tier."""
    matches = r.get("blind_identification", {}).get("matches_target", False)
    diag = r.get("diagnostic_completeness", {}).get("level", "none")
    mm = morph_mean(r)

    if not matches:
        return "hard_fail"
    if diag != "species":
        return "soft_fail"
    if mm >= STRICT_MORPH_THRESHOLD:
        return "strict_pass"
    return "borderline"


def extract_caste(filename: str) -> str | None:
    parts = filename.split("::")
    return parts[2] if len(parts) >= 4 else None


# ── Thumbnail generation ──────────────────────────────────────────────────────

def make_thumbnail_base64(image_path: Path, width: int = THUMBNAIL_WIDTH) -> str | None:
    """Load an image, resize to thumbnail, return base64-encoded JPEG."""
    if not HAS_PIL:
        return None
    if not image_path.exists():
        return None
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        ratio = width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((width, new_height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        print(f"  Warning: could not process {image_path}: {e}")
        return None


def load_thumbnail_pil(image_path: Path, width: int = 150) -> Image.Image | None:
    """Load and resize an image for contact sheet use."""
    if not HAS_PIL or not image_path.exists():
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        ratio = width / img.width
        new_height = int(img.height * ratio)
        return img.resize((width, new_height), Image.LANCZOS)
    except Exception:
        return None


# ── Enrich results ────────────────────────────────────────────────────────────

def enrich_results(results: list[dict]) -> list[dict]:
    """Add tier, morph_score, caste, blind_id_species to each result."""
    enriched = []
    for r in results:
        if "error" in r:
            continue
        entry = {
            "file": r.get("file", ""),
            "species": r.get("species", ""),
            "tier": classify_tier(r),
            "morph_score": round(morph_mean(r), 2),
            "caste": extract_caste(r.get("file", "")),
            "blind_id_species": r.get("blind_identification", {}).get("species", ""),
            "matches_target": r.get("blind_identification", {}).get("matches_target", False),
            "diag_level": r.get("diagnostic_completeness", {}).get("level", ""),
            "overall_pass": r.get("overall_pass", False),
        }
        enriched.append(entry)
    return enriched


# ── HTML generation ───────────────────────────────────────────────────────────

def build_html(entries: list[dict], image_dir: Path) -> str:
    """Build self-contained HTML viewer with base64 thumbnails."""
    # Collect unique values for filters
    all_species = sorted(set(e["species"] for e in entries))
    all_tiers = TIER_ORDER
    all_castes = sorted(set(e["caste"] for e in entries if e["caste"]))

    # Compute counts
    from collections import Counter
    tier_counts = Counter(e["tier"] for e in entries)
    species_counts = Counter(e["species"] for e in entries)

    # Build thumbnail data
    print("Generating thumbnails...")
    thumb_data = []
    for i, e in enumerate(entries):
        sp = e["species"]
        img_path = image_dir / sp / e["file"]
        b64 = make_thumbnail_base64(img_path)
        if b64 is None:
            continue
        thumb_data.append({**e, "b64": b64, "idx": i})
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(entries)} images")

    print(f"  Total thumbnails: {len(thumb_data)}")

    # Group by species for display
    by_species = {}
    for td in thumb_data:
        by_species.setdefault(td["species"], []).append(td)

    # Build cards HTML
    cards_html = []
    for sp in all_species:
        sp_entries = by_species.get(sp, [])
        if not sp_entries:
            continue
        # Sort by tier order then morph score descending
        tier_rank = {t: i for i, t in enumerate(TIER_ORDER)}
        sp_entries.sort(key=lambda x: (tier_rank.get(x["tier"], 99), -x["morph_score"]))

        cards_html.append(f'<div class="species-group" data-species="{sp}">')
        cards_html.append(f'<h2>{sp.replace("_", " ")} ({len(sp_entries)} images)</h2>')
        cards_html.append('<div class="grid">')
        for td in sp_entries:
            caste_attr = td["caste"] or "unknown"
            border_color = TIER_COLORS[td["tier"]]
            cards_html.append(
                f'<div class="card" data-species="{td["species"]}" '
                f'data-tier="{td["tier"]}" data-caste="{caste_attr}" '
                f'style="border-color: {border_color};">'
                f'<img src="data:image/jpeg;base64,{td["b64"]}" loading="lazy">'
                f'<div class="info">'
                f'<div class="filename" title="{td["file"]}">{td["file"][:30]}...</div>'
                f'<div class="tier tier-{td["tier"]}">{td["tier"]}</div>'
                f'<div>morph: {td["morph_score"]:.1f}</div>'
                f'<div>caste: {caste_attr}</div>'
                f'<div>blind ID: {td["blind_id_species"]}</div>'
                f'</div></div>'
            )
        cards_html.append('</div></div>')

    cards_joined = "\n".join(cards_html)

    # Filter buttons HTML
    species_buttons = "\n".join(
        f'<button class="filter-btn species-btn" data-filter="{sp}" '
        f'onclick="toggleFilter(\'species\', \'{sp}\')">'
        f'{sp.replace("Bombus_", "B. ")} ({species_counts[sp]})</button>'
        for sp in all_species
    )
    tier_buttons = "\n".join(
        f'<button class="filter-btn tier-btn" data-filter="{t}" '
        f'style="background:{TIER_COLORS[t]}; color: {"#000" if t in ("borderline",) else "#fff"};" '
        f'onclick="toggleFilter(\'tier\', \'{t}\')">'
        f'{t} ({tier_counts.get(t, 0)})</button>'
        for t in all_tiers
    )
    caste_buttons = "\n".join(
        f'<button class="filter-btn caste-btn" data-filter="{c}" '
        f'onclick="toggleFilter(\'caste\', \'{c}\')">'
        f'{c} ({sum(1 for e in entries if e["caste"] == c)})</button>'
        for c in all_castes
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Judge Results Viewer</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #333; }}
h2 {{ color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
.filters {{ background: #fff; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.filter-group {{ margin-bottom: 10px; }}
.filter-group label {{ font-weight: bold; display: inline-block; width: 80px; }}
.filter-btn {{ padding: 6px 12px; margin: 2px; border: 2px solid #ccc; border-radius: 4px; cursor: pointer; font-size: 13px; }}
.filter-btn.active {{ border-color: #333; font-weight: bold; box-shadow: 0 0 4px rgba(0,0,0,0.3); }}
.filter-btn:hover {{ opacity: 0.8; }}
.counts {{ background: #fff; padding: 10px 15px; border-radius: 8px; margin-bottom: 15px; font-size: 14px; color: #666; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.card {{ background: #fff; border: 4px solid #ccc; border-radius: 8px; width: 210px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); transition: opacity 0.2s; }}
.card img {{ width: 100%; display: block; }}
.card .info {{ padding: 8px; font-size: 11px; line-height: 1.4; }}
.card .filename {{ font-weight: bold; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.card .tier {{ font-weight: bold; padding: 2px 6px; border-radius: 3px; display: inline-block; margin: 2px 0; }}
.tier-strict_pass {{ background: #4CAF50; color: #fff; }}
.tier-borderline {{ background: #FFEB3B; color: #000; }}
.tier-soft_fail {{ background: #FF9800; color: #fff; }}
.tier-hard_fail {{ background: #F44336; color: #fff; }}
.card.hidden {{ display: none; }}
.species-group.hidden {{ display: none; }}
#reset-btn {{ background: #2196F3; color: #fff; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-size: 14px; }}
#reset-btn:hover {{ background: #1976D2; }}
</style>
</head>
<body>
<h1>LLM Judge Results Viewer</h1>
<p>Total images: {len(thumb_data)} | Species: {len(all_species)} | Generated from results.json</p>

<div class="filters">
  <div class="filter-group">
    <label>Species:</label>
    {species_buttons}
  </div>
  <div class="filter-group">
    <label>Tier:</label>
    {tier_buttons}
  </div>
  <div class="filter-group">
    <label>Caste:</label>
    {caste_buttons}
  </div>
  <div style="margin-top:10px;">
    <button id="reset-btn" onclick="resetFilters()">Reset All Filters</button>
    <span id="visible-count" style="margin-left:15px; color:#666;"></span>
  </div>
</div>

<div id="content">
{cards_joined}
</div>

<script>
const activeFilters = {{ species: new Set(), tier: new Set(), caste: new Set() }};

function toggleFilter(type, value) {{
    const set = activeFilters[type];
    if (set.has(value)) {{
        set.delete(value);
    }} else {{
        set.add(value);
    }}
    // Update button state
    document.querySelectorAll(`.${type}-btn`).forEach(btn => {{
        btn.classList.toggle('active', set.has(btn.dataset.filter));
    }});
    applyFilters();
}}

function resetFilters() {{
    activeFilters.species.clear();
    activeFilters.tier.clear();
    activeFilters.caste.clear();
    document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
    applyFilters();
}}

function applyFilters() {{
    let visible = 0;
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {{
        const sp = card.dataset.species;
        const tier = card.dataset.tier;
        const caste = card.dataset.caste;
        const show = (activeFilters.species.size === 0 || activeFilters.species.has(sp))
                  && (activeFilters.tier.size === 0 || activeFilters.tier.has(tier))
                  && (activeFilters.caste.size === 0 || activeFilters.caste.has(caste));
        card.classList.toggle('hidden', !show);
        if (show) visible++;
    }});
    // Hide empty species groups
    document.querySelectorAll('.species-group').forEach(group => {{
        const sp = group.dataset.species;
        const showGroup = activeFilters.species.size === 0 || activeFilters.species.has(sp);
        const hasVisible = group.querySelectorAll('.card:not(.hidden)').length > 0;
        group.classList.toggle('hidden', !showGroup || !hasVisible);
    }});
    document.getElementById('visible-count').textContent = `Showing ${{visible}} / {len(thumb_data)} images`;
}}

// Init
applyFilters();
</script>
</body>
</html>"""
    return html


# ── Contact sheet generation ──────────────────────────────────────────────────

def make_contact_sheet(
    entries: list[dict],
    image_dir: Path,
    output_path: Path,
    species: str,
    cols: int = 5,
    rows: int = 4,
    thumb_size: int = 150,
):
    """Create a PNG contact sheet with pass examples (top) and fail examples (bottom)."""
    if not HAS_PIL:
        print("  PIL not available, skipping contact sheets")
        return

    pass_entries = [e for e in entries if e["tier"] == "strict_pass"]
    fail_entries = [e for e in entries if e["tier"] in ("soft_fail", "hard_fail")]

    max_per_grid = cols * rows

    pass_sample = pass_entries[:max_per_grid]
    fail_sample = fail_entries[:max_per_grid]

    # Determine thumbnail dimensions (square-ish cells)
    cell_w = thumb_size
    cell_h = thumb_size  # will be adjusted per image

    # Load thumbnails
    def load_thumbs(sample):
        thumbs = []
        for e in sample:
            img_path = image_dir / e["species"] / e["file"]
            thumb = load_thumbnail_pil(img_path, width=cell_w)
            if thumb:
                thumbs.append((thumb, e))
        return thumbs

    pass_thumbs = load_thumbs(pass_sample)
    fail_thumbs = load_thumbs(fail_sample)

    if not pass_thumbs and not fail_thumbs:
        print(f"  No images found for {species}, skipping contact sheet")
        return

    # Compute max height across all thumbnails for uniform cell size
    all_thumbs = pass_thumbs + fail_thumbs
    max_h = max(t.height for t, _ in all_thumbs) if all_thumbs else cell_h

    # Layout: title bar + pass grid + separator + fail grid
    title_h = 40
    label_h = 30
    padding = 5
    grid_h = rows * (max_h + padding) + padding
    total_w = cols * (cell_w + padding) + padding
    total_h = title_h + label_h + grid_h + 10 + label_h + grid_h + 10

    sheet = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    # Draw using PIL only (no matplotlib needed for this)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(sheet)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf", 18)
        font_label = ImageFont.truetype("/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf", 14)
    except Exception:
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font_title = ImageFont.load_default()
            font_label = ImageFont.load_default()

    # Title
    sp_display = species.replace("_", " ")
    draw.text((padding, 8), f"{sp_display} — Judge Results Contact Sheet", fill=(0, 0, 0), font=font_title)

    def draw_grid(thumbs, y_start, label_text, label_color):
        draw.rectangle([(0, y_start), (total_w, y_start + label_h)], fill=label_color)
        draw.text((padding, y_start + 6), label_text, fill=(255, 255, 255), font=font_label)
        y_grid = y_start + label_h
        for idx, (thumb, entry) in enumerate(thumbs):
            row_i = idx // cols
            col_i = idx % cols
            if row_i >= rows:
                break
            x = padding + col_i * (cell_w + padding)
            y = y_grid + padding + row_i * (max_h + padding)
            sheet.paste(thumb, (x, y))

    # Pass grid
    y_pass = title_h
    draw_grid(pass_thumbs, y_pass, f"STRICT PASS ({len(pass_thumbs)} shown)", (76, 175, 80))

    # Fail grid
    y_fail = title_h + label_h + grid_h + 10
    draw_grid(fail_thumbs, y_fail, f"FAIL ({len(fail_thumbs)} shown)", (244, 67, 54))

    sheet.save(output_path, quality=90)
    print(f"  Saved contact sheet: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    results_path: Path = DEFAULT_RESULTS,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
):
    print(f"Loading results from {results_path}")
    data = json.loads(results_path.read_text())
    raw_results = data.get("results", [])
    entries = enrich_results(raw_results)
    print(f"  Total entries (no errors): {len(entries)}")

    all_species = sorted(set(e["species"] for e in entries))
    print(f"  Species: {all_species}")

    # Tier summary
    from collections import Counter
    tier_counts = Counter(e["tier"] for e in entries)
    for t in TIER_ORDER:
        print(f"  {t}: {tier_counts.get(t, 0)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── HTML viewer ───────────────────────────────────────────────────────────
    if not HAS_PIL:
        print("WARNING: Pillow not installed. Cannot generate thumbnails for HTML viewer.")
        print("  Install with: pip install Pillow")
    else:
        html = build_html(entries, image_dir)
        html_path = output_dir / "viewer.html"
        html_path.write_text(html)
        print(f"\nHTML viewer saved to: {html_path}")

    # ── Contact sheets ────────────────────────────────────────────────────────
    if not HAS_PIL:
        print("WARNING: Pillow not installed. Cannot generate contact sheets.")
    else:
        print("\nGenerating contact sheets...")
        for sp in all_species:
            sp_entries = [e for e in entries if e["species"] == sp]
            # Sort: passes by morph descending, fails by morph ascending
            sp_entries.sort(key=lambda x: (-1 if x["tier"] == "strict_pass" else 1, -x["morph_score"]))
            out_path = output_dir / f"contact_{sp}.png"
            make_contact_sheet(sp_entries, image_dir, out_path, sp)

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LLM judge results as HTML grid viewer and PNG contact sheets",
    )
    parser.add_argument(
        "--results", type=Path, default=DEFAULT_RESULTS,
        help=f"Path to results.json (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--image-dir", type=Path, default=DEFAULT_IMAGE_DIR,
        help=f"Directory containing species image folders (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    run(
        results_path=args.results,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
