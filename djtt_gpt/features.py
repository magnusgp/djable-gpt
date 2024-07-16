from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from djtt_gpt.config import TABLE_GPT_DIR_TRAIN

import pandas as pd
import numpy as np

app = typer.Typer()


@app.command()
def visualize_table(
    input_path: Path = TABLE_GPT_DIR_TRAIN
):
    # Load jsonl data file
    train_dataset = pd.read_json(path_or_buf="data/raw/tablegpt_data/train/train_ErrorDetection.jsonl", lines=True)
    prompt = train_dataset["prompt"][0]
    # print(prompt)
    # The table is in the prompt, right after two newlines and right after "Input:\n". There may be more than one table in the prompt.
    prompts = prompt.split("\n\n")
    tables = []
    for p in prompts:
        if p.startswith("Input:"):
            p = p.replace("Input:\n", "")
            tables.append(p)
        elif p.startswith("In:"):
            p = p.replace("In:\n", "")
            tables.append(p)
    # print(tables)
    
    for prompt in tables:
        df = convert_prompt_to_dataframe(prompt)
        print(df)
    
    # The completion is the erroneous cells
    # The erroneous cells are in the completion, right after "erroneous_cells\": \"" and before "\"}"
    completions = train_dataset["completion"]
    erroneous_cells = []
    for c in completions:
        erroneous_cells.append(c.split("erroneous_cells\": \"")[1].split("\"}")[0])
    # print(erroneous_cells)
    
def convert_prompt_to_dataframe(
    prompt: str
):
    # Split the string into lines
    lines = prompt.split('\n')

    # Extract headers
    headers = lines[0].split('|')[1:-1]  # Remove the leading and trailing '|'

    # Extract rows
    rows = [line.split('|')[1:-1] for line in lines[2:]]  # Skip the header and separator lines

    # Create a DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Replace 'nan' with NaN
    df.replace('nan', np.nan, inplace=True)
    
    return df

if __name__ == "__main__":
    app()
