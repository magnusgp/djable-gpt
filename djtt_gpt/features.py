from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from djtt_gpt.config import RAW_DATA_DIR, HF_PATH_DIR, DATASET_SUBSET

from tabulate import tabulate

from datasets import load_dataset

app = typer.Typer()


@app.command()
def main(
    input_path: Path = HF_PATH_DIR,
    output_path: Path = RAW_DATA_DIR,
    data_subset: str = DATASET_SUBSET
):
    
    train_dataset = load_dataset(f"{input_path}", f"{data_subset}", split="train", cache_dir=f"{output_path}")
    test_dataset = load_dataset(f"{input_path}", f"{data_subset}", split="test", cache_dir=f"{output_path}")
    
    train_df = train_dataset.to_pandas()
    test_df = test_dataset.to_pandas()

    logger.info(f"Train dataset shape: {train_df.shape}")
    logger.info(f"Test dataset shape: {test_df.shape}")
    
    print(tabulate(train_df.head(), headers='keys', tablefmt='psql'))

if __name__ == "__main__":
    app()
