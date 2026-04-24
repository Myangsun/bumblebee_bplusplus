#!/usr/bin/env python3
"""Build Figure 5.24 — stacked failure-chain panels for all three rare species.

D3 (d4_synthetic) has no harmed B. ashtoni at seed 42, so the ashtoni panel
is drawn from D4 (d5_llm_filtered) harmed chains. The B. sandersoni and
B. flavidus panels come from D3 harmed chains. Caption should reflect this.
"""
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

ROOT = Path("/home/msun14/bumblebee_bplusplus")
OUT = ROOT / "docs/plots/failure/chains_failure_3species"

PANELS = [
    ("B. ashtoni (harmed under D4)",
     ROOT / "docs/plots/failure/chains_d5_harmed/gallery/ashtoni__Bombus_ashtoni5828709435.png"),
    ("B. sandersoni (harmed under D3)",
     ROOT / "docs/plots/failure/chains_d4_harmed/gallery/sandersoni__Bombus_sandersoni4952898591.png"),
    ("B. flavidus (harmed under D3)",
     ROOT / "docs/plots/failure/chains_d4_harmed/gallery/flavidus__Bombus_flavidus4512075898.png"),
]

fig, axes = plt.subplots(3, 1, figsize=(14, 9))
for ax, (title, path) in zip(axes, PANELS):
    assert path.exists(), f"missing {path}"
    ax.imshow(Image.open(path))
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, loc="left", y=1.01)
fig.suptitle("Figure 5.24 — Representative failure chains for the three rare species",
             fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(str(OUT) + ".png", dpi=200, bbox_inches="tight")
fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}.png and .pdf")
