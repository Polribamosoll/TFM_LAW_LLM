# Setup --------------------------------------------------------------

# General libraries
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone
import torch
import time
import yaml

# Local assets
from functions.etl_functions import *

# Load env
load_dotenv()

# Parameters --------------------------------------------------------------

# Load parameters from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Pinecone
pinecone = Pinecone(api_key = os.environ.get('PINECONE_API_KEY'))

# Connect
index_name = 'lawgpt-unstructured-db'
index = pinecone.Index(index_name)

# Existinc IDs
existing_ids = index.list()

# Embed model
embed_model = HuggingFaceEmbeddings(
    model_name = config["embedding_model"],
    model_kwargs = {'device': device},
    encode_kwargs = {'device': device, 'batch_size': 32}
)

# Input dir
input_dir = "prepared_data/boe_year/"

# Batch size
batch_size = config['batch_size']

# Function definition --------------------------------------------------------------

# Load function
def load(input_dir, batch_size, embed_model, index):

    # Start timing the run
    start_time = time.time()

    # Show message
    print("Starting upload of BOE documents")

    # Count the total number of boe_year folders
    total_years = sum(1 for year_folder in os.listdir(input_dir) if year_folder.startswith("boe_"))

    # Create a tqdm progress bar for the years
    with tqdm(total=total_years, desc='Processing years') as pbar_years:
        # Main loop
        for year_folder in os.listdir(input_dir):
            if year_folder.startswith("boe_"):
                year_path = os.path.join(input_dir, year_folder)
                
                # Count the number of CSV files in the year folder
                num_files = len([f for f in os.listdir(year_path) if f.endswith('.csv')])
                total_batches = (num_files + batch_size - 1) // batch_size

                # Create a tqdm progress bar for the batches in the year
                with tqdm(total=total_batches, desc=f'Processing batches in {year_folder}') as pbar_batches:
                    # Load all CSV files in the current year folder
                    dfs = []
                    for csv_file in os.listdir(year_path):
                        if csv_file.endswith(".csv"):
                            csv_path = os.path.join(year_path, csv_file)
                            dfs.append(pd.read_csv(csv_path))
                    
                    # Concatenate loaded DataFrames
                    df_combined = pd.concat(dfs, ignore_index=True)
                    
                    # Split the combined DataFrame into batches and process each batch
                    for i in range(0, len(df_combined), batch_size):
                        batch = df_combined.iloc[i:i+batch_size]
                        embed_and_insert_data(batch, embed_model, index)
                        # Update the progress bar for each batch processed
                        pbar_batches.update(1)

                 # Update the progress bar for each year processed
                pbar_years.update(1)

    # Show message
    print("Upload of BOE documents completed")

    # Runtime

    # End time of notebook run
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Convert elapsed time to hours and minutes
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)

    # Print the result
    print(f"Time elapsed: {hours} hours and {minutes} minutes.")

# Run function

# Main
if __name__=='__main__':
    # Call the load function
    load(input_dir, batch_size, embed_model, index)
    