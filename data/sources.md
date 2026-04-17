# Dataset Sources

All input data used in this pipeline. Keep this file updated whenever new seed footage is added so the dataset can be fully replicated.

---

## Category 1 — Urban Aerial (VisDrone2019-VID-train)

**License:** Academic / non-commercial research use  
**Citation:** VisDrone Challenge, Tianjin University  
**Download:** https://drive.google.com/file/d/1NSNapZQHar22OYzQYuXCugA3QlMndzvw/view?usp=sharing

After download, the archive contains video frames as sequential images (not MP4s).

```bash
# Extract
unzip VisDrone2019-VID-train.zip -d VisDrone2019-VID-train/

# Structure: VisDrone2019-VID-train/sequences/<seq_name>/<frame_number>.jpg
# Convert to MP4s:
python scripts/dataset_prep.py --sources visdrone
```

### Selected Sequences (9 clips)

| Sequence | Scene |
|---|---|
| uav0000288_00001_v | urban intersection, crosswalks, heavy mixed traffic |
| uav0000278_00001_v | urban arterial, multi-lane commercial road |
| uav0000244_01440_v | mixed urban corridor, residential + commercial |
| uav0000248_00001_v | roundabout |
| uav0000357_00920_v | commercial intersection, bikes + cars |
| uav0000239_03720_v | residential street, narrow, tight traffic |
| uav0000323_01173_v | road + parking lot |
| uav0000307_00000_v | urban expressway, trucks + cars, city skyline |
| uav0000308_00000_v | pedestrian-heavy urban intersection |

**Output:** `seed_videos_prepped/<seq_name>.mp4` — 93 frames @ 16fps, 1280×720, H.264 CRF 18  
**Annotations:** `seed_videos_prepped/annotations/<seq_name>.txt` — VisDrone format, frame indices remapped to 1–93

---

## Category 2 — Fire / Smoke Aerial (Pexels)

**License:** [Pexels Free License](https://www.pexels.com/license/) — free for commercial use, no attribution required  
**Download script:** `scripts/download_fire_seeds.sh`  
**Raw files:** `fire_raw/` (gitignored)

To re-download manually: visit each URL below, click **Free Download → HD**.

| Canonical name | Pexels URL | Author | Downloaded filename |
|---|---|---|---|
| wildfire_smoke_rising | https://www.pexels.com/video/aerial-view-of-forest-fire-with-smoke-rising-30937710/ | Kelly | 12773167_1280_720_24fps.mp4 |
| farm_fire_south_africa | https://www.pexels.com/video/aerial-view-of-farm-fire-in-south-africa-33661465/ | Kelly | 13228892_1280_720_30fps.mp4 |
| forest_fire_dense_smoke | https://www.pexels.com/video/aerial-view-of-forest-fire-through-dense-smoke-29702246/ | Kelly | ⚠ 14302118_720_1280_24fps.mp4 — PORTRAIT (720×1280), needs re-download at landscape orientation |
| structure_fire_buildings | https://www.pexels.com/video/drone-footage-of-a-burning-establishments-8365989/ | Alan W | 8365989-hd_1280_720_30fps.mp4 |

> Note: Pexels download filenames use an internal video-file ID that differs from the page URL ID.  
> The canonical names above are the authoritative identifiers for this pipeline.

**Planned domain shifts from fire/smoke seeds:**  
`structure_fire`, `wildfire`, `car_crash_smoke`, `gunfire_scene`, `flooding`, `dust_storm`, `night_fire`
