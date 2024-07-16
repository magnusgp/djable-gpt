from pathlib import Path

import typer

from datasets import load_dataset

from djtt_gpt.config import RAW_DATA_DIR, HF_PATH_DIR, DATASET_SUBSET, TABLE_GPT_DIR_TRAIN

import pandas as pd

app = typer.Typer()

@app.command()
def main(
    input_path: Path = HF_PATH_DIR,
    output_path: Path = RAW_DATA_DIR,
    data_subset: str = DATASET_SUBSET
):
    
    train_dataset = load_dataset(f"{input_path}", f"{data_subset}", split="train", cache_dir=f"{output_path}")
    test_dataset = load_dataset(f"{input_path}", f"{data_subset}", split="test", cache_dir=f"{output_path}")
    
if __name__ == "__main__":
    app()
