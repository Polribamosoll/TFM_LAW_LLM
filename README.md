# Law LLM

# Objective

The objective is to use LLM models for text generation enriched with law & judiciary information from Spanish public institutions

# Structure

## Data Sources

Official sources:

- BOE: Boletín Oficial del Estado - Código Penal

## Folders

- functions: ETL and App specific functions
- auxiliar: Auxiliar functions
- logs: Logs of ETL evolution
- models: Temporary models saved here
- notebooks: Notebooks directory
- prepared_data: Prepared data for models
- raw_data: Data extracted from main sources
- save_folder: Temporary folder for model weights
- scripts: Auxiliar scripts for ETL scheduling
- usage: Directory for app usage tracking files


Other:

## Main files

- config.yaml: Code configuration
- notebooks

## Development notebooks

### Data Pipeline

- 1_extract_data.ipynb: Data extraction from sources
- 2_transform_data.ipynb: Prepare datasets for encoding
- 3_encode_save_data.ipynb: Encode & save prepared datasets

### Similarity

- 4_similarity_search: Evaluate similarity search methods

### Models

- 5_hybrid_rag_model.ipynb: Test methods to provide context to LLM
- 6_deploy_model.ipynb: Model used for App deployment
- 7_fine_tuning.ipynb: Fine-Tuning model

## Other

- .gitignore: Ignore files for Git
- functions.py: Auxiliar functions
- requirements.txt: Libraries used in project

## Platforms

- Computing: Gradient Paperspace
- Vector database: Pinecone

License: MIT

June 2024, Barcelona
