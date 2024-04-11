
# Setup

# Platform

## Register on Paperspace-Gradient

https://www.paperspace.com/artificial-intelligence

## Upgrade account

Pro-Account: Mid tier level GPU's ~ 10 eur/month

## Create Notebooks

Create a project and use only one notebook
Start the notebook with the option: Start from sratch
All notebooks shut down after X hours (6 recommended)
File changes are only saved in the notebooks folder (default directory)
Python libraries are not saved

# Conenction with GitHub

## HTTPS approach (easy):

One time setting:

- git config --global user.name "user_name"
- git config --global user.email "user_email"
- git remote set-url origin https://user_name(yours):git_access_token@github.com/user_name(owner)/repo_name.git

## SSH approach

Need to be done each time machine is restarted

- ssh-keygen -t ed25519 -C “user@example.com”
- cat ~/.ssh/id_ed25519.pub
- eval "$(ssh-agent -s)"
- ssh-add ~/.ssh/id_ed25519

# Environment variables

Keys for Pinecone, HF are stored in .env file, editable in GitHub:

Process: mv .env secrets.env -> Edit -> mv secrets.env .env

- PINECONE_ENVIRONMENT: To access vector DB
- PINECONE_API_KEY: To access vector DB
- HF_KEY: Relevant to log in HuggingFace

To retrieve keys:

from dotenv import load_dotenv
load_dotenv()
os.environ.get('PINECONE_ENVIRONMENT')

# Dependencies

Run 0_machine_setup.ipynb every time the machine restarts

# Run the project

Gitignore prevents CSV from being pushed to Git
As of this, all files need to be created using the notebooks locally

# Python helpers

$ python --version
Python 3.10.0
$ python -m pip install ipykernel
$ python -m ipykernel install --user
Installed kernelspec python3 in /Users/soma/Library/Jupyter/kernels/python3
