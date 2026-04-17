from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_summary(results: list[dict], output_file: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df = df.sort_values(by=["accuracy", "f1_score"], ascending=False)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return df
