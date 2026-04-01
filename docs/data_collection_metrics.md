# GBIF Data Collection Metrics

## Collection Parameters

- **Library:** bplusplus (`bplusplus.collect`)
- **API:** GBIF Occurrence Search (`https://api.gbif.org/v1/occurrence/search`) via `pygbif`
- **Max per species:** 3,000 (random sample from all valid occurrences)
- **Date collected:** February 2026

## Quality Filters Applied

| Filter | Value | Effect |
|---|---|---|
| `mediaType` | `StillImage` | Only photographs (no audio, video) |
| `basisOfRecord` | `HUMAN_OBSERVATION`, `MACHINE_OBSERVATION`, `OBSERVATION` | Field observations only — excludes museum specimens (`PRESERVED_SPECIMEN`), fossils, DNA samples, living specimens (zoo/garden) |
| `lifeStage` | `Adult` | Adults only — excludes larvae, pupae, eggs |
| `occurrenceStatus` | `PRESENT` | Confirmed presence records only |
| `year` | `2010–2025` | Recent records only (better image quality) |
| URL validation | valid image URL in `media[0]` | Drops records without downloadable images |

## Downloaded Image Counts

| Species | Images | Notes |
|---|---|---|
| Bombus impatiens | 3,000 | Cap reached (common species) |
| Bombus griseocollis | 3,000 | Cap reached |
| Bombus bimaculatus | 3,000 | Cap reached |
| Bombus ternarius | 3,000 | Cap reached |
| Bombus pensylvanicus | 3,000 | Cap reached |
| Bombus rufocinctus | 2,127 | |
| Bombus perplexus | 1,546 | |
| Bombus fervidus | 1,506 | |
| Bombus terricola | 1,028 | |
| Bombus borealis | 992 | |
| Bombus vagans | 901 | |
| Bombus citrinus | 804 | |
| Bombus affinis | 566 | Federally endangered (SARA/ESA) |
| Bombus flavidus | 314 | Cuckoo bumblebee (social parasite, naturally rare) |
| **Bombus sandersoni** | **80** | **Rare; few GBIF records pass filters** |
| **Bombus ashtoni** | **36** | **Cuckoo bumblebee, critically rare** |
| **Total** | **21,900** | |

## Why Rare Species Have Few Images

The three target species for synthetic augmentation (B. sandersoni, B. ashtoni, B. affinis) are genuinely rare:

1. **Small populations** — fewer field encounters → fewer iNaturalist uploads
2. **Quality filters compound scarcity** — requiring `Adult` + `StillImage` + `year >= 2010` drops a large fraction of what little exists
3. **Museum specimens excluded** — `basisOfRecord` filter removes pinned-specimen photographs, which for rare/declining species may be a significant portion of available imagery

## Potential Filter Adjustments

- **Include `PRESERVED_SPECIMEN`:** Museum collections have extensive imagery of rare Bombus species (e.g., USDA ARS, AMNH). Could substantially increase B. sandersoni and B. ashtoni counts, but images look very different from field photos (pinned, dorsal view, white background).
- **Remove `lifeStage: Adult`:** Many GBIF records lack life stage metadata entirely and are dropped. Removing this filter would recover unlabeled-but-adult records at the cost of some non-adult contamination.
- **Expand year range:** Pre-2010 records exist but tend to have lower resolution images.
