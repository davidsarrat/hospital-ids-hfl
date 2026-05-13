from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_NAME = "hospital-ids-hfl"
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_PATH = Path("data/processed/cicids_clean.parquet")
DEFAULT_PARTITIONS_DIR = Path("data/partitions")
DEFAULT_SHARED_DIR = Path("shared")
DEFAULT_SCALER_PATH = DEFAULT_SHARED_DIR / "preprocessing" / "scaler.json"


@dataclass(frozen=True)
class Hospital:
    hospital_id: str
    region: str
    preferred_attack_groups: tuple[str, ...]


HOSPITALS: tuple[Hospital, ...] = (
    Hospital("hospital_eu_01", "region_eu", ("BENIGN", "DOS")),
    Hospital("hospital_eu_02", "region_eu", ("BENIGN", "BRUTE_FORCE", "WEB")),
    Hospital("hospital_eu_03", "region_eu", ("BENIGN", "PORTSCAN")),
    Hospital("hospital_na_01", "region_na", ("BENIGN", "DDOS")),
    Hospital("hospital_na_02", "region_na", ("BENIGN", "BOTNET", "INFILTRATION")),
    Hospital(
        "hospital_na_03",
        "region_na",
        (
            "BENIGN",
            "DOS",
            "BRUTE_FORCE",
            "WEB",
            "PORTSCAN",
            "DDOS",
            "BOTNET",
            "INFILTRATION",
            "HEARTBLEED",
            "OTHER_ATTACK",
        ),
    ),
)

REGIONS: tuple[str, ...] = ("region_eu", "region_na")


def hospitals_by_region(region: str) -> list[Hospital]:
    hospitals = [hospital for hospital in HOSPITALS if hospital.region == region]
    if not hospitals:
        raise ValueError(f"Unknown region: {region}")
    return hospitals


def parse_regions(value: str | None) -> list[str]:
    if not value:
        return list(REGIONS)
    regions = [region.strip() for region in value.split(",") if region.strip()]
    unknown = sorted(set(regions) - set(REGIONS))
    if unknown:
        raise ValueError(f"Unknown regions: {', '.join(unknown)}")
    return regions


def partition_dir(hospital_id: str, partitions_dir: Path = DEFAULT_PARTITIONS_DIR) -> Path:
    return partitions_dir / hospital_id


def region_checkpoint(region: str, round_number: int, shared_dir: Path = DEFAULT_SHARED_DIR) -> Path:
    return shared_dir / "checkpoints" / region / f"round_{round_number}.pt"


def global_checkpoint(round_number: int, shared_dir: Path = DEFAULT_SHARED_DIR) -> Path:
    return shared_dir / "checkpoints" / "global" / f"round_{round_number}.pt"
