import os
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import NotebookExporter

# List of notebook files
notebooks = [
    "0_machine_setup.ipynb",
    "1_extract_data.ipynb",
    "2_transform_data.ipynb",
    "3_encode_save_data.ipynb"
]

# Execute notebooks
for notebook_file in notebooks:
    # Set up execution parameters
    ep = ExecutePreprocessor(timeout=None)
    notebook_name, _ = os.path.splitext(notebook_file)
    
    # Execute notebook
    with open(notebook_file, 'r') as f:
        nb = ep.preprocess(f, {'metadata': {'path': './'}})
    
    # Save executed notebook
    with open(f'{notebook_name}_executed.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

# Export notebooks to Python scripts
for notebook_file in notebooks:
    notebook_name, _ = os.path.splitext(notebook_file)
    exporter = NotebookExporter()
    output_notebook, _ = exporter.from_filename(f'{notebook_name}_executed.ipynb')
    
    # Save Python script
    with open(f'{notebook_name}_executed.py', 'w', encoding='utf-8') as f:
        f.write(output_notebook)
