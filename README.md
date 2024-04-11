# Berto

# Objective

The objective is to use LLM models for text generation enriched with law & judiciary information from Spanish public institutions

# Structure

## Data Sources

Official sources:

- BOE: Boletín Oficial del Estado
- BORME: Boletín Oficial del Registro Mercantil

Other sources (WIP):

- CENDOJ: Centro de Documentación Judicial
- Law books
- Law exams

## Folders

- streamlit: Streamlit settings
- apps: Streamlit applications for product testing
- backup: Backup of old notebooks
- functons: ETL and App specific functions
- auxiliar: Auxiliar functions
- logs: Logs of ETL evolution
- models: Temporary models saved here
- notebooks: Notebooks directory
- prepared_data: Prepared data for models
    - splitted_input_method.csv: Data extracted raw_data and splitted in chunks
- raw_data: Data extracted from main sources
- save_folder: Temporary folder for model weights
- scripts: Auxiliar scripts for ETL scheduling
- usage: Directory for app usage tracking files
- venv: Virtual environment

Other:

- optimum-nvidia: Repo for optimum LLM run with Nvidia
- raptor: Repo for RAPTOR RAG methods

## Setup

Run %pip install -r libraries.txt to install required libraries for notebooks
Run %pip install -r requirements.txt to install required libraries for app

## Main files

- config.yaml: App configuration
- run_etl: Run ETL process
- app.py: Main Streamlit app

## Deployment files

- Procfile: Set up and run app
- runtime.txt: Python version
- setup.sh: Creates config.toml

## Development notebooks

### Data Pipeline

- 1_extract_data.ipynb: Data extraction from sources
- 2_transform_data.ipynb: Prepare datasets for encoding
- 3_encode_save_data.ipynb: Encode & save prepared datasets

### Similarity

- 4_similarity_search: Evaluate similarity search methods

### Models

- 5_hybrid_rag_model.ipynb: Test methods to provide context to LLM
- 6_deploy_model.ipynb: Final RAG model used for App

## Other

- .gitignore: Ignore files for Git
- functions.py: Auxiliar functions
- requirements.txt: Libraries used in project

## Platforms

- Computing: Saturn Cloud
- Vector database: Pinecone
- Database: Heroku
- Deploy: Heroku

License: MIT

January 2024, Barcelona
