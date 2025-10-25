"""
GBIF Data Collection for Massachusetts Bumblebee Species
Focus on rare species: Bombus terricola and Bombus fervidus

Based on the Bumble Bees of New England field guide and your research proposal
"""

import bplusplus
from pathlib import Path

# Define output directory
GBIF_DATA_DIR = Path("./GBIF_MA_BUMBLEBEES")

# Massachusetts bumblebee species based on your field guides
# Marking rare/at-risk species with comments
ma_bumblebee_species = [

    # 'Bombus_vagans_Smith',
    # 'Bombus_ternarius_Say'
    # Common/Abundant species (A)
    "Bombus_impatiens",        # Common Eastern Bumble Bee - Most abundant
    "Bombus_griseocollis",     # Brown-belted Bumble Bee - Abundant
    "Bombus_bimaculatus",      # Two-spotted Bumble Bee - Abundant

    # Species at Risk (SP) - YOUR PRIMARY TARGETS
    "Bombus_terricola",        # Yellow-banded Bumble Bee - RARE (SP, HE)
    "Bombus_fervidus",         # Golden Northern Bumble Bee - RARE (SP, LH)

    # Other species present in MA
    "Bombus_ternarius_Say",        # Tricolored Bumble Bee - Abundant
    "Bombus_borealis",         # Northern Amber Bumble Bee - Abundant
    "Bombus_rufocinctus",      # Red-belted Bumble Bee
    "Bombus_vagans_Smith",           # Half Black Bumble Bee
    "Bombus_sandersoni",       # Sanderson's Bumble Bee (low habitat, rare)
    "Bombus_perplexus",        # Perplexing Bumble Bee

    # Parasitic/Cuckoo species
    "Bombus_citrinus",         # Lemon Cuckoo-Bumble Bee
    "Bombus_flavidus",         # Fernald's Cuckoo Bumble Bee

    # Likely Extirpated in MA (according to your proposal)
    "Bombus_pensylvanicus",    # American Bumble Bee - LIKELY EXTIRPATED
    "Bombus_affinis",          # Rusty Patched Bumble Bee - LIKELY EXTIRPATED
    "Bombus_ashtoni",          # Ashton's Cuckoo-Bumble Bee - LIKELY EXTIRPATED
]

# Define search parameters
# Adding geographic constraint for Massachusetts
search = {
    "scientificName": ma_bumblebee_species,
    "stateProvince": "Massachusetts",  # Filter for MA only
    "country": "US"  # United States
}

print(
    f"Collecting GBIF data for {len(ma_bumblebee_species)} bumblebee species in Massachusetts...")
print(f"Target rare species: Bombus terricola and Bombus fervidus\n")

# Run collection
# Downloading 2000 images per species (you can adjust this)
bplusplus.collect(
    group_by_key=bplusplus.Group.scientificName,
    search_parameters=search,
    images_per_group=2000,  # Download more than needed for rare species
    output_directory=GBIF_DATA_DIR,
    num_threads=5
)

print("\n" + "="*60)
print("Collection complete!")
print(f"Data saved to: {GBIF_DATA_DIR}")
print("="*60)
print("\nNext steps:")
print("1. Run bplusplus.prepare() to organize the data")
print("2. Analyze species distribution, especially B. terricola and B. fervidus")
print("3. Run bplusplus.train() and bplusplus.test()")
