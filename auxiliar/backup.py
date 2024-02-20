# Get BOE URLs
def get_boe_urls_backup(start_date, end_date, boe_class_list, filename):

    # Create an empty list to store the days excluding Sundays
    days_excluding_sundays = []

    # Iterate through the dates and add them to the list if they are not Sundays
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() != 6:  # Sunday corresponds to 6 in the weekday() function
            days_excluding_sundays.append(current_date)
        current_date += datetime.timedelta(days=1)  # Increment the date by one day

    # Check if the file exists and get the last ID from it
    try:
        with open(filename, 'r') as f:
            last_id = f.readlines()[-1].strip().split(',')[0]
    except FileNotFoundError:
        with open(filename, 'w') as f:
            f.write('day_id,url,title\n')
        last_id = f'{start_date.year}{"{:02d}".format(start_date.month)}{"{:02d}".format(start_date.day)}'

    # Initialize a DataFrame to store the scraped data
    data_df = pd.DataFrame(columns=['id_', 'urlxml', 'titulo'])

    # Total days for log
    total_days = len(days_excluding_sundays)
    days_processed = 0

    # Iterate through the days and scrape data for each day
    for date_ in days_excluding_sundays:
        
        # Format id
        id_ = f'{date_.year}{"{:02d}".format(date_.month)}{"{:02d}".format(date_.day)}'
        
        # Continue with last id
        if id_ <= last_id:
            continue
        
        # Prepare df
        data_df = pd.DataFrame(columns=['id_', 'urlxml', 'titulo'])
        url_ = f'https://www.boe.es/diario_boe/xml.php?id=BOE-S-{id_}'
        
        try:
            response = requests.get(url_)
            if response.status_code != 200:
                continue
            content = response.content
            soup = BeautifulSoup(content, 'html.parser')
            item_elements = soup.find_all('item')
            
            for item_element in item_elements:
                titulo = item_element.find('titulo').text
                urlxml = item_element.find('urlxml').text if item_element.find('urlxml') else None
                if urlxml is None:
                    continue
                if urlxml[27] not in ['A', 'C']:
                    continue
                data_df = data_df.append({'id_': id_, 'urlxml': f'https://www.boe.es{urlxml}', 'titulo': titulo}, ignore_index=True)
                
            with open(filename, 'a') as f:
                data_df.to_csv(f, header=False, index=False)
                
        except Exception as e:
            logging.error(f"Error processing URL: {url_} - Error: {str(e)}")
            
        days_processed += 1
        progress_percentage = (days_processed / total_days) * 100
        logging.info(f"Progress: {progress_percentage:.2f}% ({days_processed}/{total_days} days processed)")
        