# Libraries ----------------------------------------------

# App version
version = "App version: Berto v0.0.2"

# General libraries
from dotenv import load_dotenv
from deta import Deta
import os

# Environment
load_dotenv(".env")
DETA_KEY = os.getenv("DETA_KEY")

# Database ----------------------------------------------

# Initialize with a project key
deta = Deta(DETA_KEY)

# Users DB
db_users = deta.Base("users_db")

# Usage DB
db_usage = deta.Base("usage_db")

# User database ----------------------------------------------

# Insert user
def insert_user(username, name, email, password, ):
    """Returns the user on a successful user creation, otherwise raises and error"""
    return db_users.put({"key": username, "name": name, "email": email, "password": password})

# Fetch all users
def fetch_all_users():
    """Returns a dict of all users"""
    res = db_users.fetch()
    return res.items

# Fetch user
def get_user(username):
    """If not found, the function will return None"""
    return db_users.get(username)

# Update user
def update_user(username, updates):
    """If the item is updated, returns None. Otherwise, an exception is raised"""
    return db_users.update(updates, username)

# Delete user
def delete_user(username):
    """Always returns None, even if the key does not exist"""
    return db_users.delete(username)

# Usage database ----------------------------------------------

# Insert usage
def insert_usage(username, time, prompt, context, tokens):
    """Returns the user on a successful user creation, otherwise raises and error"""
    return db_usage.put({"key": username + '-' + time, "username": username, "time": time, "prompt": prompt, "context": context, "tokens": tokens})
