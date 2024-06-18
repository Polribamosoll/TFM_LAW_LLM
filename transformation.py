# Setup --------------------------------------------------------------

# General libraries needed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
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

# Input dir
input_dir = "raw_data/boe_year/"

# Output dir
output_dir = "prepared_data/boe_year/"

# Maximum length of a text
max_chunk_size = config['max_chunk_size']

# Chunk overlap
chunk_overlap_size = config['chunk_overlap']

# Separators
separators = ["\n\n", "\n", ". ", " ", ""]

# Splitter
recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = max_chunk_size,
    chunk_overlap  = chunk_overlap_size,
    length_function = len,
    separators = separators
)

# Splitter function
def splitter(text):
    chunks = recursive_text_splitter.split_text(text)
    return chunks

# Function definition --------------------------------------------------------------

# Transform function
def transform(input_dir, output_dir, splitter):

    # Start timing the run
    start_time = time.time()

    # Show message
    print("Starting transformation of BOE documents")

    # Count the total number of boe_year folders
    total_years = sum(1 for year_folder in os.listdir(input_dir) if year_folder.startswith("boe_"))

    # Length of years
    print("Total years to be processed: ", total_years)

    # Create a tqdm progress bar for the years
    with tqdm(total=total_years, desc='Processing years') as pbar_years:
        # Loop by folder
        for year_folder in sorted(os.listdir(input_dir)):
            if not year_folder.startswith("boe_"):
                continue
            year = year_folder.split('_')[1]
            year_input_path = os.path.join(input_dir, year_folder)
            year_output_path = os.path.join(output_dir, year_folder)
            if not os.path.isdir(year_input_path):
                continue
            
            # Create the output directory if it doesn't exist
            os.makedirs(year_output_path, exist_ok=True)
            
            # Iterate over each CSV file in the current year folder
            for csv_file in sorted(os.listdir(year_input_path)):
                # Check if csv files
                if not csv_file.endswith(".csv"):
                    continue
                csv_input_path = os.path.join(year_input_path, csv_file)
                
                # Check if file exists
                flag = False
                for file_ in os.listdir(year_output_path):
                    if file_.startswith(csv_file.split(".")[0]):
                        flag = True
                        break
                if flag:
                    continue

                # Read file
                df = pd.read_csv(csv_input_path)
                
                try:
                    # Apply the splitter function to the text column
                    text_chunks = splitter(df['text'].iloc[0])
                except TypeError:
                    continue
                
                # Create a new DataFrame with individual chunks and unique identifiers
                for i, chunk in enumerate(text_chunks):
                    chunk_df = pd.DataFrame([{
                        'id': df['id'].iloc[0],
                        'url': df['url'].iloc[0],
                        'title': df['title'].iloc[0],
                        'legislative_origin': df['legislative_origin'].iloc[0],
                        'department': df['department'].iloc[0],
                        'rang': df['rang'].iloc[0],
                        'text_id': f"{df['id'].iloc[0]}_chunk{i+1}",
                        'text': chunk
                    }])

                    csv_output_path = os.path.join(year_output_path, f'{csv_file.split(".")[0]}_{i+1:04}.csv')
                    
                    # Write the DataFrame to the output directory
                    chunk_df.to_csv(csv_output_path, index=False)
        
        # Update the progress bar for each year processed
        pbar_years.update(1)

        # Print a message when each year is completed
        print(f"Year {year} completed.")

    # Show message
    print("Transformation of BOE documents completed")
    
    # Runtime

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
    # Call the transform function
    transform(input_dir, output_dir, splitter)
    
