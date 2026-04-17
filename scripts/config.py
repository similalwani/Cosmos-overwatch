from pathlib import Path

BASE = Path(__file__).parent.parent

# Scene labels are the authoritative record for each sequence.
SELECTED = [
    ("uav0000288_00001_v", "urban intersection, crosswalks, heavy mixed traffic"),
    ("uav0000278_00001_v", "urban arterial, multi-lane commercial road"),
    ("uav0000244_01440_v", "mixed urban corridor, residential + commercial"),
    ("uav0000248_00001_v", "roundabout"),
    ("uav0000357_00920_v", "commercial intersection, bikes + cars"),
    ("uav0000239_03720_v", "residential street, narrow, tight traffic"),
    ("uav0000323_01173_v", "road + parking lot"),
    ("uav0000307_00000_v", "urban expressway, trucks + cars, city skyline"),
    ("uav0000308_00000_v", "pedestrian-heavy urban intersection"),
]

SEQ_NAMES = [name for name, _ in SELECTED]

# Cosmos Transfer2.5 native: 93 frames @ 16fps = 5.8125s
OUTPUT_FRAMES  = 93
DST_FPS        = 16
SRC_FPS        = 30
DURATION_SEC   = OUTPUT_FRAMES / DST_FPS

SEQ_DIR  = BASE / "VisDrone2019-VID-train" / "sequences"
ANN_DIR  = BASE / "VisDrone2019-VID-train" / "annotations"
PREP_DIR = BASE / "seed_videos_prepped"

FIRE_RAW_DIR = BASE / "fire_raw"
FIRE_NAMES = [
    "farm_fire_south_africa",
    "forest_fire_dense_smoke",
    "structure_fire_buildings",
    "wildfire_smoke_rising",
]
# "forest_fire_dense_smoke_PORTRAIT" excluded (portrait orientation)
