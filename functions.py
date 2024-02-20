# General libraries
import pandas as pd
import numpy as np
import datetime
import time
import csv
import os

# Scrapping
from bs4 import BeautifulSoup
import unicodedata
import unidecode
import requests
import string
import json
import lxml
import xml
import bs4
import gc
import re

# Langchain
from langchain.text_splitter import CharacterTextSplitter

# Multiprocessing
from multiprocessing import Pool, Queue, Process
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value, Lock
from functools import partial

# Cloud
from google.cloud import storage

# Other
from tqdm.notebook import tqdm
import logging

# Local
from functions import *

# Warnings
import warnings
warnings.filterwarnings("ignore")

# Auxiliar --------------------------------------------------------------

def list_csv_files(folder_path):
    
    # List CSV files
    return [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# 1. Extract BOE/BORME --------------------------------------------------------------

# Get XML from URL
def get_xml(url):
    
    # Get XML
    headers = {'accept': 'application/xml;q=0.9, */*;q=0.8'}
    response = requests.get(url, headers=headers)
    return response.text

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
    
    # Initialize logging
    logging_file = 'boe_id_extraction.log'
    if os.path.exists(logging_file):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Read existing data from the previous file, if it exists
    if os.path.exists(filename):
        data_df = pd.read_csv(filename)
    else:
        data_df = pd.DataFrame(columns=['day_id', 'url', 'title'])

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                logging.warning('File found, getting last day extracted')
                # Extracts the last ID if there is more than one line (excluding the header)
                last_id = lines[-1].strip().split(',')[0]
            elif len(lines) == 1:
                # If there's only one line (header), set last_id based on start_date
                logging.warning('File found with only headers, starting extraction from start_date')
                last_id = f'{start_date.year}{"{:02d}".format(start_date.month)}{"{:02d}".format(start_date.day)}'
            else:
                # No lines in the file, implying it's newly created or emptied
                logging.warning('File found without lines, writing headers and starting extraction')
                with open(filename, 'w') as f:
                    f.write('day_id,url,title\n')
                last_id = f'{start_date.year}{"{:02d}".format(start_date.month)}{"{:02d}".format(start_date.day)}'
    except FileNotFoundError:
        logging.warning('File not found, writing headers and starting extraction')
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
            response = requests.get(url_)
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
                data_list.append({'day_id': id_, 'url': f'https://www.boe.es{url_xml}', 'title': title})
            
            # Append to data_df_temp
            data_df_temp = pd.concat([data_df_temp, pd.DataFrame(data_list)], ignore_index=True)
            
            # Save in CSV
            with open(filename, 'a') as f:
                data_df_temp.to_csv(f, header=False, index=False)
                
        except Exception as e:
            logging.error(f"Error processing URL: {url_} - Error: {str(e)}")
        
        # Log progress
        days_processed += 1
        progress_percentage = (days_processed / total_days) * 100
        logging.info(f"Progress: {progress_percentage:.2f}% ({days_processed}/{total_days} days already processed)")

        # Concatenate data_df_temp with data_df after each iteration
        data_df = pd.concat([data_df, data_df_temp], ignore_index=True)
    
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

# Extract BOE by year
def extract_boe_year(data_directory, df, year, log_queue):
    
    # Initialize logging
    logging_file = 'boe_year_extraction.log'
    if os.path.exists(logging_file):
        os.remove(logging_file)
    logging.basicConfig(filename=logging_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Format df year
    df_year = df.loc[df['day_id'].astype(str).str.startswith(year)]
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

    # Initialize tqdm progress bar
    pbar = tqdm(total=len(df_year), desc=f"Year {year}")
    
    # Initialize count of processed IDs
    processed_ids = 0

    # Loop through each row in the DataFrame
    for index, row in df_year.iterrows():
        url = row['url']
        id_ = int(url.split('-')[-1])
        
        # If url already extracted, skip
        if id_ <= last_id_year:
            processed_ids += 1
            pbar.update(1)
            continue
        
        day_id = row['day_id']
        title = row['title']

        # Access info and save it
        try:
            xml = get_xml(url)
            origen_legislativo = get_legislativo(xml)
            departamento = get_departamento(xml)
            rango = get_rango(xml)
            text = get_text(xml)
            record = {
                'id': day_id,
                'url': url,
                'title': title,
                'legislative_origin': origen_legislativo,
                'department': departamento,
                'rang': rango,
                'text': text
            }
            df_row = pd.DataFrame(record, index=[0])[cols]
            with open(output_filename, 'a') as f:
                df_row.to_csv(f, header=False, index=False)
            
            # Update processed IDs count
            processed_ids += 1
            
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error processing URL: {url} - HTTP Error: {str(http_err)}")
        
        # Update tqdm progress
        pbar.update(1)

        # Log progress
        progress_percentage = processed_ids / len(df_year) * 100
        log_queue.put(f"Year {year} - Progress: {progress_percentage:.2f}% ({processed_ids}/{len(df_year)} IDs processed)")

    # Close tqdm
    pbar.close()
    
def setup_logger(log_queue):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    while True:
        record = log_queue.get()
        if record is None:
            break
        logger.info(record)
    
# 2. Transformation functions --------------------------------------------------------------

# Recursive text splitter (v0)
def recursive_text_splitter_backup(text, chunk_size, separators):
    
    chunks = []
    current_separator_index = 0
    
    for separator in separators:
        
        current_separator_index += 1
        
        if len(text) < chunk_size:
            if len(text) > 0:
                chunks.append(text)
            break
            
        # Break condition, no separator, split by chunk_size
        if separator == "":
            chunks += [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            break
            
        if separator in text:
            while separator in text:
                if len(text) < chunk_size:
                    chunks.append(text)
                    text = ""
                    break
                split_at = text.rfind(separator, 0, chunk_size)
                # Chunk is too big, try next separator
                if split_at == -1:
                    chunk, text = text.split(separator, 1)
                    chunks += recursive_text_splitter(chunk, chunk_size, separators[current_separator_index:])
                else:
                    chunks.append(text[:split_at+len(separator)])
                    text = text[split_at+len(separator):]
                    
    return chunks

# 3. Encoding functions --------------------------------------------------------------

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

# Google Buckets --------------------------------------------------------------

# Function to download file from Google Cloud Storage
def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    