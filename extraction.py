# Setup --------------------------------------------------------------

# General libraries
from multiprocessing import cpu_count
from dotenv import load_dotenv
import datetime
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

# BOE IDs filename
boe_ids_filename = "raw_data/boe_ids.csv"

# Path to store full extraction
folder_path = "raw_data/boe_year/"

# BOE Class
boe_class = ["A", "C"]

# Start date
start_date = datetime.date(1978, 1, 1)

# Define the end date to extract
end_date = datetime.datetime.now().date()

# Years to extract
years = [str(i) for i in range(1978, 2025)]

# Number of cpu's
num_cpus = cpu_count() - 4

# Function definition --------------------------------------------------------------

# Extract function
def extract(boe_ids_filename, folder_path, boe_class, start_date, end_date, years, num_cpus):

    # Start timing the run
    start_time = time.time()

    # Run BOE IDs

    # Show message
    print("Starting extraction of BOE IDs")

    # Update BOE IDs
    boe_ids = extract_boe_ids(start_date, end_date, boe_class, boe_ids_filename)

    # Filter empty titles
    boe_filtered_ids = boe_ids.dropna(subset=['title'])

    # Show message
    print("BOE IDs extraction completed")

    # Run BOE Extraction

    # Show message
    print("Starting extraction of BOE documents")

    # Show
    print("Number of CPUs: ", num_cpus)

    # Run extraction in parallel
    extract_boe_year_parallel(folder_path, boe_filtered_ids, years, num_cpus)

    # Show message
    print("BOE documents extraction completed")

    # Runtime

    # End time of notebook run
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Convert elapsed time to hours and minutes
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)

    # Print the result
    print(f"Time elapsed: {hours} hours and {minutes} minutes.")

# Run function --------------------------------------------------------------

# Main
if __name__=='__main__':
    # Call the extract function
    extract(boe_ids_filename, folder_path, boe_class, start_date, end_date, years, num_cpus)
    