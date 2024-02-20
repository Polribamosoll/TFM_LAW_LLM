# LawGPT

# Objective

The objective is to use LLM models for text generation enriched with law & judiciary information from Spanish public institutions

# Structure

## Data Sources

Official sources:

- BOE: Boletín Oficial del Estado
- BORME: Boletín Oficial del Registro Mercantil
- CENDOJ: Centro de Documentación Judicial

Other sources (WIP):

- Law books
- Law exams

## Folders

- apps: Streamlit applications for product testing
- auxiliar: Auxiliar functions
- models: Temporary models saved here
- prepared_data: Prepared data for models
    - splitted_input_method.csv: Data extracted raw_data and splitted in chunks
- raw_data: Data extracted from sources
- save_folder: Temporary folder for model weights
- scripts: Auxiliar scripts

## Setup

Run %pip install -r requirements.txt to install required libraries

## Notebooks

### Data Pipeline

- 1_extract_data.ipynb: Data extraction from sources
- 2_transform_data.ipynb: Prepare datasets for encoding
- 3_encode_save_data.ipynb: Encode & save prepared datasets

### Similarity

- 4_similarity_search: Evaluate similarity search methods

### Models

- 5_hybrid_rag_model.ipynb: Test RAG techniques
- 6_run_rag_model.ipynb: Run final RAG model

### Deploy

- 7_deploy_model.ipynb: Deploy model using FastAPI

## Other

- .gitignore: Ignore files for Git
- functions.py: Auxiliar functions
- requirements.txt: Libraries used in project

## Platforms

- Computing: Gradient Notebooks
- Vector database: Pinecone
- Storage: Google Buckets

License: MIT

January 2024, Barcelona
