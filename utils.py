# General libraries
from bs4 import BeautifulSoup
from typing import List
import pandas as pd
import numpy as np
import datetime
import unicodedata
import requests
import tiktoken
import logging
import sys
import re
import os

# Multiprocessing
from tqdm.contrib.concurrent import process_map
from multiprocessing import cpu_count

# Other
from tqdm.notebook import tqdm

# Warnings
import warnings
warnings.filterwarnings("ignore")

# Auxiliar --------------------------------------------------------------

# List CSV files in path
def list_csv_files(folder_path):
    # List CSV files
    return [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Read csv
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

# 1. Extract BOE/BORME --------------------------------------------------------------

# Get XML from URL
def get_xml(url, timeout_param=60):
    try:
        response = requests.get(url, timeout=timeout_param)
        if response.status_code != 200:
            # If the response status code is not 200 (OK)
            # return None to indicate failure
            return None
        return response.text
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occur during the request
        logging.info(f"Error getting XML for URL {url}: {str(e)}")
        return None
    
# Get text from XML
def get_text(XML):
    # Get text
    text_all = ""
    soup = BeautifulSoup(XML, "html.parser") 
    text = soup.select_one("documento > texto").get_text()
    return text

# Get title from XML
def get_title(XML):
    # Get title
    soup = BeautifulSoup(XML, "html.parser")
    title = soup.find("titulo")
    return title.get_text()

# Get legislativo from XML
def get_legislativo(XML):
    # Get legislativo
    soup = BeautifulSoup(XML, "html.parser")
    origen_legislativo = soup.find("origen_legislativo")
    return origen_legislativo.get_text()

# Get departamento from XML
def get_departamento(XML):
    # Get departamento
    soup = BeautifulSoup(XML, "html.parser")
    departamento = soup.find("departamento")
    return departamento.get_text()

# Get rango from XML
def get_rango(XML):
    # Get rango
    soup = BeautifulSoup(XML, "html.parser")
    rango = soup.find("rango")
    return rango.get_text()

# Extract BOE IDs
def extract_boe_ids(start_date, end_date, boe_letters, filename):
    # Disable existing logging configuration
    logging.getLogger().handlers = []
    
    # Remove propagation to root logger to prevent logging to screen
    logging.getLogger().propagate = False
    
    # Initialize logging inside the function
    logging_file = 'logs/boe_id_extraction.log'
    if os.path.exists(logging_file):
        os.remove(logging_file)
        
    # Create the file if it doesn't exist
    open(logging_file, 'a').close()
    
    # Log config
    logging.basicConfig(
        filename=logging_file, 
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Read existing data from the previous file, if it exists
    if os.path.exists(filename):
        data_df = pd.read_csv(filename)
    else:
        data_df = pd.DataFrame(columns=['day_id', 'url', 'title'])

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                logging.info('File found, getting last day extracted')
                # Extracts the last ID if there is more than one line (excluding the header)
                last_id = lines[-1].strip().split(',')[0]
            elif len(lines) == 1:
                # If there's only one line (header), set last_id based on start_date
                logging.info('File found with only headers, starting extraction from start_date')
                last_id = f'{start_date.year}{"{:02d}".format(start_date.month)}{"{:02d}".format(start_date.day)}'
            else:
                # No lines in the file, implying it's newly created or emptied
                logging.info('File found without lines, writing headers and starting extraction')
                with open(filename, 'w') as f:
                    f.write('day_id,url,title\n')
                last_id = f'{start_date.year}{"{:02d}".format(start_date.month)}{"{:02d}".format(start_date.day)}'

    except FileNotFoundError:
        logging.info('File not found, writing headers and starting extraction')
        with open(filename, 'w') as f:
            f.write('day_id,url,title\n')
        last_id = f'{start_date.year}{"{:02d}".format(start_date.month)}{"{:02d}".format(start_date.day)}'
        
    # Create an empty list to store the days excluding Sundays
    days_excluding_sundays = []

    # Iterate through the dates and add them to the list if they are not Sundays
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() != 6:  # Sunday corresponds to 6 in the weekday() function
            days_excluding_sundays.append(current_date)
        current_date += datetime.timedelta(days = 1) # Increment the date by one day
        
    # Total days for logging
    total_days = len(days_excluding_sundays)
    days_processed = 0

    # Prepare DataFrame
    data_df_temp = pd.DataFrame(columns=['day_id', 'url', 'title'])

    # Iterate through the days and scrape data for each day
    for date_ in tqdm(days_excluding_sundays, desc="Processing days"):  # Wrapped with tqdm for progress tracking
        
        # Format id
        id_ = f'{date_.year}{"{:02d}".format(date_.month)}{"{:02d}".format(date_.day)}'
        
        # Continue with last id
        if id_ <= last_id:
            days_processed += 1
            continue

        # Generate again df temp
        data_df_temp = pd.DataFrame(columns=['day_id', 'url', 'title'])
        url_ = f'https://www.boe.es/diario_boe/xml.php?id=BOE-S-{id_}'
        
        try:
            response = requests.get(url_, timeout = 60)
            if response.status_code != 200:
                continue
            content = response.content
            soup = BeautifulSoup(content, 'html.parser')
            item_elements = soup.find_all('item')
            
            # Loop by item
            data_list = []
            for item_element in item_elements:
                title = item_element.find('titulo').text
                url_xml = item_element.find('urlxml').text if item_element.find('urlxml') else None
                if url_xml is None or url_xml[27] not in boe_letters:
                    continue
                # Append output
                data_list.append({'day_id': id_, 'url': f'https://www.boe.es{url_xml}', 'title': title})
            
            # Append to data_df_temp
            data_df_temp = pd.concat([data_df_temp, pd.DataFrame(data_list)], ignore_index=True)
            
            # Save in CSV
            with open(filename, 'a') as f:
                data_df_temp.to_csv(f, header=False, index=False)
                
        except Exception as e:
            logging.info(f"Error processing URL: {url_} - Error: {str(e)}")
        
        # Log progress
        days_processed += 1
        progress_percentage = (days_processed / total_days) * 100
        logging.info(f"Progress: {progress_percentage:.2f}% ({days_processed}/{total_days} days already processed)")

        # Concatenate data_df_temp with data_df after each iteration
        data_df = pd.concat([data_df, data_df_temp], ignore_index=True)
    
    # Return
    return data_df

# Find titles with keywords
def find_titles_with_keywords(df, key_words):
     # Normalize keywords and titles to lowercase and remove accents
    normalized_keywords = [unicodedata.normalize('NFD', keyword).encode('ascii', 'ignore').decode('utf-8').lower() for keyword in key_words]
    
    # Convert 'title' column values to strings and then normalize
    df['normalized_title'] = df['title'].apply(lambda x: unicodedata.normalize('NFD', str(x)).encode('ascii', 'ignore').decode('utf-8').lower())

    # Use regular expressions to match keywords
    pattern = '|'.join(map(re.escape, normalized_keywords))
    filtered_df = df[df['normalized_title'].str.contains(pattern, flags=re.IGNORECASE, na=False)]
    
    # Reset index and drop the old index
    filtered_df = filtered_df.reset_index(drop=True)
    
    # Drop the 'normalized_title' column
    filtered_df = filtered_df.drop(columns=['normalized_title'])
    
    # Return the filtered and reset DataFrame
    return filtered_df

# Process year
def process_year(year, df, data_directory):
    # Get index from input
    df_year = df.loc[df['day_id'].astype(str).str.startswith(str(year))]
    df_year['index_id'] = df_year['url'].str.split('-').str[-1].astype(int)
    df_year = df_year.sort_values(by='index_id').reset_index(drop=True)
    
    # If output file exists, retrieve last checked, else write it with header
    output_filename = os.path.join(data_directory, f'boe_{year}.csv')
    cols = ['id', 'url', 'title', 'legislative_origin', 'department', 'rang', 'text']
    try:
        last_url = pd.read_csv(output_filename)['url'].iloc[-1]
        last_id_year = int(last_url.split('-')[-1])
    except (FileNotFoundError, IndexError):
        with open(output_filename, 'w') as f:
            f.write(f'{",".join(cols)}\n')
        last_id_year = 0
    
    # Initialize count of processed IDs
    processed_ids = 0

    # Loop through each row in the DataFrame
    for index, row in df_year.iterrows():
        url = row['url']
        id_ = int(url.split('-')[-1])
        
        # If url already extracted, skip
        if id_ <= last_id_year:
            processed_ids += 1
            continue
        
        # Day ID and title
        day_id = row['day_id']
        title = row['title']

        # Access info and save it
        try:
            xml = get_xml(url)

            # Handle possible failures of individual functions
            try:
                origen_legislativo = get_legislativo(xml)
            except Exception as e:
                logging.info(f"Error getting legislative origin for URL {url}: {str(e)}")
                origen_legislativo = None

            try:
                departamento = get_departamento(xml)
            except Exception as e:
                logging.info(f"Error getting department for URL {url}: {str(e)}")
                departamento = None

            try:
                rango = get_rango(xml)
            except Exception as e:
                logging.info(f"Error getting range for URL {url}: {str(e)}")
                rango = None

            try:
                text = get_text(xml)
            except Exception as e:
                logging.info(f"Error getting text for URL {url}: {str(e)}")
                text = None

            # Prepare record
            record = {
                'id': day_id,
                'url': url,
                'title': title,
                'legislative_origin': origen_legislativo,
                'department': departamento,
                'rang': rango,
                'text': text
            }

            # Create DataFrame row
            df_row = pd.DataFrame(record, index=[0])[cols]

            # Append row to CSV
            with open(output_filename, 'a') as f:
                df_row.to_csv(f, header=False, index=False)

            # Update processed IDs count
            processed_ids += 1

        except requests.exceptions.RequestException as req_err:
            logging.info(f"Error processing URL {url}: {str(req_err)}")
        except Exception as e:
            logging.info(f"Unexpected error processing URL {url}: {str(e)}")
        
        # Log progress within the year
        progress_percentage = processed_ids / len(df_year) * 100
        logging.info(f"Year {year} - Progress: {progress_percentage:.2f}% ({processed_ids}/{len(df_year)} IDs processed)")

        # Print in console
        if processed_ids % 1000 == 0:
            print(f"Year {year} - Progress: {progress_percentage:.2f}% ({processed_ids}/{len(df_year)} IDs processed)")

# Process year wrapper
def process_year_wrapper(args):
    return process_year(*args)  
    
# Extract boe in parallel
def extract_boe_year_parallel(data_directory, df, years, num_cpus=None):
    # Disable existing logging configuration
    logging.getLogger().handlers = []
    
    # Remove propagation to root logger to prevent logging to screen
    logging.getLogger().propagate = False
    
    # Initialize logging inside the function
    logging_file = 'logs/boe_extraction.log'
    if os.path.exists(logging_file):
        os.remove(logging_file)
        
    # Create the file if it doesn't exist
    open(logging_file, 'a').close()
    
    # Log config
    logging.basicConfig(
        filename=logging_file, 
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Determine the number of CPUs to use if not specified
    if num_cpus is None:
        num_cpus = cpu_count()

    # Use tqdm's concurrent process_map for parallel processing
    process_map(process_year_wrapper, [(year, df, data_directory) for year in years], max_workers=num_cpus)
    
# 2. Transformation functions --------------------------------------------------------------

# NLTK splitter
def split_text_nltk(text, chunk_size):
    words = nltk.word_tokenize(text, language = 'spanish') 
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# spaCy splitter
# nlp = spacy.load("es_core_news_sm")
def split_text_spacy(text, chunk_size):
    doc = nlp(text)
    chunks = []
    chunk = []
    for token in doc:
        if len(chunk) + 1 > chunk_size:
            chunks.append(' '.join(chunk))
            chunk = []
        chunk.append(token.text)
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks

# Generate summary with LLM
def generate_summary(pipeline, tokenizer, prompt, context):
    # Format the prompt for summarization
    messages = [
        {"role": "user", "content": f"{prompt}: {context}:"}
    ]
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    # Generate the summary using the pipeline
    outputs = pipeline(prompt, do_sample = True, top_p = 0.95, top_k = 50)
    # Extract the text after '<start_of_turn>model\n'
    output_text = outputs[0]["generated_text"][len(prompt):]
    # Return
    return output_text

# Count tokens
def count_tokens(string: str, encoding_name: str = "cl100k_base") -> int:
    # Get encoding from tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    # Encode the string using the specified encoding
    encoded_string = encoding.encode(string)
    # Count the number of tokens
    num_tokens = len(encoded_string)
    return num_tokens

# Filter context by tokens
def filter_context_by_tokens(input, context_key, max_model_tokens, count_tokens):
    # Initialize cumulative token count
    cumulative_tokens = 0

    # Filtered list to store dictionaries
    filtered_context = []

    # Iterate through the list of dictionaries
    for item in input:
        # Get the context value based on the context_key
        context_value = item[context_key]
        
        # Calculate number of tokens for 'context' value
        token_count = count_tokens(context_value)
        
        # Cumulative sum of token counts
        cumulative_tokens += token_count
        
        # Check if cumulative tokens are still less than max_model_tokens
        if cumulative_tokens < max_model_tokens:
            filtered_context.append(item)
        else:
            break
    
    return filtered_context

# Function to calculate tokens processed per second
def calculate_tokens_per_second(num_tokens, elapsed_time):
    tokens_per_second = num_tokens / elapsed_time
    return tokens_per_second

# 3. Encoding functions --------------------------------------------------------------

# Get embedding function for GTE
def get_embedding(text: str) -> list[float]:
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []
    embedding = embedding_model.encode(text)

    return embedding.tolist()

# Filter df rows by byte count
def filter_df_by_byte_count(df, max_bytes):
    # Check if 'text' column exists in the DataFrame
    if 'text' not in df.columns:
        print("Error: 'text' column not found in the DataFrame.")
        return None

    # Calculate the number of bytes for each 'text' and create a new column
    df['num_bytes'] = df['text'].apply(lambda x: sys.getsizeof(x.encode()))

    # Filter the DataFrame based on the 'num_bytes' column
    filtered_df = df[df['num_bytes'] < max_bytes]

    # Reset the index to have continuous indices if needed
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df

# Clean text for embeddings
def clean_text(text):
    text = text.replace('\n', ' ')
    text = text.replace('|', ' ')
    text = ' '.join(text.split())  # Remove excessive whitespace
    return text

# Define a function to fetch vectors in batches
def fetch_vectors_in_batches(index, unique_ids, batch_size=1000):
    ids = []
    embeddings = []
    num_ids = len(unique_ids)
    for i in range(0, num_ids, batch_size):
        batch_ids = unique_ids[i:i+batch_size]
        vectors = index.fetch(ids=batch_ids)
        for id, vector in vectors['vectors'].items():
            ids.append(id)
            embeddings.append(vector['values'])
    return ids, embeddings

# RAG functions --------------------------------------------------------------

# Re-rank documents function
def rank_documents(cross_encoder, text_field, query:str, retrieved_documents:List[dict]):
    """
    Ranks retrieved documents based on their relevance to a given query using a cross-encoder model.

    Parameters:
    - cross_encoder (CrossEncoder): A cross-encoder model from the sentence-transformers library.
    - query (str): The query string for which the documents are to be ranked.
    - retrieved_documents (List[dict]): A list of dictionaries representing documents. Each dictionary should have a 'text' field
      containing the document text and any additional fields that you want to retain in the output dictionary.

    Returns:
    - dict: A dictionary where the key is the rank position (starting from 0 for the most relevant document)
      and the value is a dictionary containing the document text and any additional fields. The documents are ranked
      in descending order of relevance to the query.

    Usage:
    ranked_docs = rank_documents(cross_encoder, query, retrieved_documents)

    Note: This function requires the sentence-transformers library and a pretrained cross-encoder model.
    """
    pairs = [[query, doc[text_field]] for doc in retrieved_documents]
    scores = cross_encoder.predict(pairs)
    ranks = np.argsort(scores)[::-1]
    ranked_docs = {rank_num: {text_field: retrieved_documents[rank_num][text_field], **retrieved_documents[rank_num]} for rank_num in ranks}
    return ranked_docs

# Google Buckets --------------------------------------------------------------

# Function to download file from Google Cloud Storage
def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

# Fine Tuning

# Preprocessing function
def preprocess_function(examples, tokenizer):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=512,
        truncation="only_second",  
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offset_mapping = inputs.pop("offset_mapping").tolist()
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        start_char = examples["answers"][i]["answer_start"]
        end_char = start_char + len(examples["answers"][i]["text"])

        start_token_idx = None
        end_token_idx = None

        for j, (start, end) in enumerate(offsets):
            if start <= start_char < end:
                start_token_idx = j
            if start < end_char <= end:
                end_token_idx = j
            if start_token_idx is not None and end_token_idx is not None:
                break

        if start_token_idx is None or end_token_idx is None:
            start_token_idx = 0
            end_token_idx = 0

        start_positions.append(start_token_idx)
        end_positions.append(end_token_idx)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs