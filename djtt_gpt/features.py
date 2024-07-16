from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from djtt_gpt.config import TABLE_GPT_DIR_TRAIN

import pandas as pd
import numpy as np

from transformers import BertTokenizer, BertModel
import torch

def create_table_from_json(
    input_path: Path = TABLE_GPT_DIR_TRAIN
):
    # Load jsonl data file
    train_dataset = pd.read_json(path_or_buf="data/raw/tablegpt_data/train/train_ErrorDetection.jsonl", lines=True)
    
    n_rows = 1
    
    prompt_list = train_dataset["prompt"]
    # The table is in the prompt, right after two newlines and right after "Input:\n". 
    for prompt in prompt_list[:n_rows]:
        prompts = prompt.split("\n\n")
        tables = []
        for p in prompts:
            if p.startswith("Input:"):
                p = p.replace("Input:\n", "")
                tables.append(p)
            elif p.startswith("In:"):
                p = p.replace("In:\n", "")
                tables.append(p)
        
        for prompt_str in tables:
            df = convert_prompt_to_dataframe(prompt_str)
            # Only return the first df for now
            return df
    
    # The completion is the erroneous cells
    completion_list = train_dataset["completion"]
    erroneous_cells = []
    for completion in completion_list[:n_rows]:
        erroneous_cells.append(completion.split("erroneous_cells\": \"")[1].split("\"}")[0])
    
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

def table_to_embeddings(
    df: pd.DataFrame,
):
    # Initialize BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    # Transform DataFrame rows to text
    texts_df = convert_df_to_text(df)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = generate_embeddings(texts_df, tokenizer, model)

    # Display embedding stats
    logger.info(f"Embeddings shape: {embeddings.shape}")

# Function to transform table row to text based on transformation option

def transform_col_to_text(col, transformation_option="title-colname-stat-col"):
    if transformation_option == "title-colname-stat-col":
        text = f"Column information: {col}"
    else:
        text = f"Column information: {col}"
    return text

def convert_df_to_text(
    df: pd.DataFrame
):
    text_output = []
    
    for column in df.columns:
        column_header = column
        column_content = ", ".join(df[column].astype(str).tolist())
        text_output.append(f'{{"Column header: {column_header}, Column content: {column_content}"}}')
    
    return text_output

# Function to generate embeddings
def generate_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
    return embeddings

if __name__ == "__main__":
    df = create_table_from_json()
    table_to_embeddings(df)
