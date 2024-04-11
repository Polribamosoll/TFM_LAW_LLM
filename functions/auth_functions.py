# Libraries ----------------------------------------------

# App version
version = "App version: Berto v0.0.2"

# Local assets
import functions.db_functions as db_functions

# General libraries
from dotenv import load_dotenv
import streamlit as st
import bcrypt

# Functions ----------------------------------------------

# Sign in function
def sign_in():
    # Fetch users in DB
    users = db_functions.fetch_all_users()
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")

    # If user use button
    if st.button("Iniciar sesión", type = "primary"):

        # Loop for user name and check password
        for user in users:
            if username == user['name'] and bcrypt.checkpw(password.encode(), user['password'].encode()):
                # Save user
                st.session_state["username"] = user['name']
                # Success message
                st.success("Inicio de sesión correcto")
                # Return
                return True
        else:
            # Return False
            return False

# Register function
def register():
    # Text input space
    new_username = st.text_input("Usuario")
    new_name = st.text_input("Nombre")
    new_email = st.text_input("Email")
    new_password = st.text_input("Contraseña", type = "password")

    # Fetch database
    users = db_functions.fetch_all_users()

    # If user use button
    if st.button("Completar registro", type = "primary"):
        # Loop by user
        for user in users:
            if new_username == user['key'] or new_email == user['email']:
                st.warning("Usuario existente")
                return False
        # Hash passwords
        hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())
        # Upload to DB
        db_functions.insert_user(new_username, new_name, new_email, hashed_password.decode())
        # Save user
        st.session_state["username"] = new_username
        # Success message
        st.success("Registro completado")
        # Return
        return True
    else:
        # Return False
        return False
