"""
Cell 0 equivalent — load raw train/test csvs.

Mirrors basic_check.ipynb cell 0 exactly (data loading only; the original
notebook is left untouched). Saves the raw frames to artifacts/ so every
later numbered script can run standalone without re-reading the csvs.
"""
import pickle
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

train = pd.read_csv(ROOT / "csvs" / "train.csv")
test = pd.read_csv(ROOT / "csvs" / "test.csv")

print(f"Train Shape: {train.shape}")
print(f"Test Shape:  {test.shape}")
print(f"Base Readmission Rate: {train['readmitted_30d'].mean():.4f}")

with open(ARTIFACTS / "train.pkl", "wb") as f:
    pickle.dump(train, f)
with open(ARTIFACTS / "test.pkl", "wb") as f:
    pickle.dump(test, f)

print(f"Saved raw train/test to {ARTIFACTS}")
