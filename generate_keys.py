
# Libraries
import streamlit_authenticator as stauth

# Local assets
import functions.db_functions as db

# Names, users and passwords
usernames = ['demo']
names = ['demo']
emails = ['demo@demo.com']
passwords = ['demo_001']

# Hash passowrds
hashed_passwords = stauth.Hasher(passwords).generate()

# Upload to DB
for (username, name, email, hash_password) in zip(usernames, names, emails, hashed_passwords):
    db.insert_user(username, name, email, hash_password)
