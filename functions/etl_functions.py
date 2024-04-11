
# Setup --------------------------------------------------------------

# General libraries
import pandas as pd
import datetime
import warnings
import logging
import os

# Scrapping
from bs4 import BeautifulSoup
import requests

# Multiprocessing
from tqdm.contrib.concurrent import process_map

# Other
from tqdm import tqdm

# Warnings
warnings.filterwarnings("ignore")

# Auxiliar --------------------------------------------------------------

# List CSV files in path
def list_csv_files(folder_path):
    # List CSV files
    return [file for file in os.listdir(folder_path) if file.endswith('.csv')]

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
    output_path = os.path.join(data_directory, f'boe_{year}')
    os.makedirs(output_path, exist_ok=True)
    cols = ['id', 'url', 'title', 'legislative_origin', 'department', 'rang', 'text']
    try:
        last_id_year = int(sorted(os.listdir(output_path))[-1].split('_')[-1].split('.')[0])
    except IndexError:
        last_id_year = 0
    
    # Initialize count of processed IDs
    processed_ids = 0

    # Loop through each row in the DataFrame
    for _, row in df_year.iterrows():
        url = row['url']
        id_ = int(url.split('-')[-1])

        
        # If url already extracted, skip
        if id_ <= last_id_year:
            processed_ids += 1
            continue
        
        url_ = url.split('-')
        output_filename = os.path.join(output_path, f'{url_[-2]}_{int(url_[-1]):06}.csv')

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

            # To csv
            df_row.to_csv(output_filename, index=False)

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
        num_cpus = os.cpu_count()

    # Use tqdm's concurrent process_map for parallel processing
    process_map(process_year_wrapper, [(year, df, data_directory) for year in years], max_workers=num_cpus)

# 3. Encoding functions --------------------------------------------------------------

# Function to embed and insert data
def embed_and_insert_data(batch, embed_model, index):
    # Convert potential float values to string in text-related columns
    batch['id'] = batch['id'].astype(str)
    batch['url'] = batch['url'].astype(str)
    batch['title'] = batch['title'].astype(str)
    batch['date'] = batch['date'].astype(str)
    batch['legislative_origin'] = batch['legislative_origin'].astype(str)
    batch['department'] = batch['department'].astype(str)
    batch['rang'] = batch['rang'].astype(str)
    batch['text_id'] = batch['text_id'].astype(str)
    batch['text'] = batch['text'].astype(str)
    
    # Text Ids formatting
    text_ids = batch['text_id'].tolist()
    texts = batch['text'].tolist()
    
    # Embed texts
    embeds = embed_model.embed_documents(texts)
    
    # Get metadata to store in DB
    metadata = [
        {'id': str(x['id']),
         'url': str(x['url']),
         'title': str(x['title']),
         'date': str(x['date']),
         'legislative_origin': str(x['legislative_origin']),
         'department': str(x['department']),
         'rang': str(x['rang']),
         'text_id': str(x['text_id']),
         'text': str(x['text'])
        } for _, x in batch.iterrows()
    ]
    
    # Add to Pinecone
    data_to_upsert = zip(text_ids, embeds, metadata)
    
    # Upsert to Pinecone
    index.upsert(data_to_upsert)