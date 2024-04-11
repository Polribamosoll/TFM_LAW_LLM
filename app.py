# Libraries ----------------------------------------------

# App version
version = "App version: Berto v0.0.2"

# Local assets
from functions.auth_functions import *
from functions.app_functions import *
from functions.db_functions import *

# General libraries
import pinecone
import yaml
import torch
import datetime
import os

# Langchain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Pinecone
from langchain_pinecone import PineconeVectorStore  
from pinecone import Pinecone

# Application
import streamlit as st
from groq import Groq
# Setup ----------------------------------------------

# Env variables
from dotenv import load_dotenv

# Load Env
load_dotenv()

# Setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load parameters from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Get parameters
embedding_model_id = config["embedding_model"]
use_quantization = config["use_quantization"]
model_id = config["model"]
pre_prompt = config["pre_prompt"]
prompt_context = config["prompt_context"]
top_k_docs = config['top_k_docs']

# App params
app_model = config["app_model"]
app_top_k_docs = config["app_top_k_docs"]
app_max_new_tokens = config["app_max_new_tokens"]
app_temperature = config["app_temperature"]
app_max_memory = config["app_max_memory"]

# Usage folder
usage_folder_path = "usage/"

# Model configuration ----------------------------------------------

# Define model details
models = {
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"}
}

# Template
chat_template = pre_prompt + prompt_context + "A continuación se proporciona el contexto para respondel al usuario: "

# Embeddings ----------------------------------------------

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name = embedding_model_id,
    model_kwargs = {'device': device},
    encode_kwargs = {'device': device, 'batch_size': 32}
) 

# Pinecone ----------------------------------------------

# Pinecone connection
pinecone = Pinecone(api_key = os.environ.get('PINECONE_API_KEY'))
index_name = os.environ.get('PINECONE_INDEX')
index = pinecone.Index(index_name)

# Vector store
vectorstore = PineconeVectorStore(index, embedding_model, "text")  

# LLM Pipeline ----------------------------------------------

# Client
client = Groq(
    api_key = os.environ.get('GROQ_API_KEY'),
)

# App configuration ----------------------------------------------

# Subheader
subheader_text = "Herramienta impulsada por inteligencia artificial diseñada para proporcionar orientación sobre asuntos legales en España"

# Footnote
disclaimer_text = """Esta aplicación tiene fines informativos únicamente y no debe ser considerada como asesoramiento legal. 
Los usuarios deben buscar el consejo de profesionales legales calificados con respecto a sus preguntas y preocupaciones legales específicas"""

# Current content
current_content = "Contenido actual: Código penal y publicaciones relacionadas"

# Streamlit app ----------------------------------------------

# Page config
st.set_page_config(page_title = "Berto", page_icon = ":libra:", layout = "wide")

# Página principal
def main():

    # Initiate logged_in
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    # If logged_in
    if st.session_state["logged_in"]:

        # Bienvenido
        st.header(f"Bienvenid@, {st.session_state['username']} :hugging_face:")

        # Header
        st.header("**Berto**: Agente de IA especialista en leyes de España :libra:")

        # Sub header
        st.subheader(subheader_text, anchor = False)

        # Footnote
        st.write(disclaimer_text, anchor = False)

        # Sub header
        st.write(current_content, anchor = False)

        # Sub header
        st.write(version, anchor = False)

        # Divider
        st.divider()

        # Initialize chat history and selected model
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Model selection
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = None

        # Box selection
        model_option = st.selectbox(
            "Selecciona un modelo:",
            options = list(models.keys()),
            format_func = lambda x: models[x]["name"],
            index = 0
        )

        # Detect model change and clear chat history if model has changed
        if st.session_state.selected_model != model_option:
            st.session_state.messages = []
            st.session_state.selected_model = model_option

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Prompt logic
        if prompt := st.chat_input("Envia una pregunta..."):

            # Initiate token count
            token_count = 0

            # Append message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Show user avatar and prompt
            with st.chat_message("user", avatar='👨‍💻'):
                st.markdown(prompt)

            # Retrieve context
            context =  retrieve_context(
                prompt, 
                app_top_k_docs,
                vectorstore
            )

            # Filter max tokens
            filtered_context = filter_context_by_tokens(
                context, 
                "context",
                app_max_new_tokens, 
                count_tokens
            )

            # Get prompt
            context_ready = str(chat_template) + str(filtered_context)

            # Try response
            try:
                # Chat completion function
                chat_completion = client.chat.completions.create(
                    model = model_option,
                    messages = [
                        {
                            "role": "system",
                            "content": context_ready
                        }
                    ] + [
                        {
                            "role": m["role"],
                            "content": m["content"]
                        }
                        for m in st.session_state.messages
                    ], 
                    max_tokens = app_max_new_tokens,
                    stream = True
                )

                # Use the generator function with st.write_stream
                with st.chat_message("assistant", avatar = "🤖"):
                    # Call generate_chat_responses function
                    chat_responses_generator = generate_chat_responses(chat_completion)
                    # Write stream
                    full_response = st.write_stream(chat_responses_generator)
            
            # Error exception
            except Exception as e:
                st.error(e, icon = "🚨")

            # Append the full response to session_state.messages
            if isinstance(full_response, str):
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response})
            else:
                # Handle the case where full_response is not a string
                combined_response = "\n".join(str(item) for item in full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": combined_response})
                
            # Layout the buttons in two columns
            cols = st.columns(5)
            
            # Button 1: Provide source
            with cols[0]:
                st.button('Proporcionar fuentes de información', on_click = provide_context_details, args = (filtered_context,), type = 'primary')

            # Button 2: Reset memory
            with cols[1]:
                st.button('Eliminar historial de chat', on_click = clear_chat_history, type = "primary")

            # Calculate total tokens
            total_tokens = count_tokens(str(prompt) + str(filtered_context))

            # Update total tokens count
            token_count = token_count + total_tokens

            # Insert usage
            insert_usage(st.session_state["username"], str(datetime.datetime.now()), str(prompt), str(full_response), total_tokens)

    # If not logged in, select an option
    else:
        selected_option = st.selectbox('Iniciar sesión / Registro', ['Entrar', 'Registro'])
        if selected_option == 'Registro':
            registration_successful = register()
            st.session_state["logged_in"] = registration_successful
            if registration_successful:
                st.rerun()
        elif selected_option == 'Entrar':
            logged_in = sign_in()
            st.session_state["logged_in"] = logged_in
            if logged_in:
                st.rerun()

# Main
if __name__ == "__main__":
    main()
