# Libraries ----------------------------------------------

# App version
version = "App version: Berto v0.0.2"

# General libraries
from dotenv import load_dotenv
from typing import Generator
import streamlit as st
import datetime as dt
import tiktoken

# Functions ----------------------------------------------

# Count tokens
def count_tokens(string: str, encoding_name: str = "cl100k_base") -> int:
    # Get encoding from tiktoken
    encoding = tiktoken.get_encoding(encoding_name)

    # Encode the string using the specified encoding
    encoded_string = encoding.encode(string)

    # Count the number of tokens
    num_tokens = len(encoded_string)

    # Return
    return num_tokens

# Filter context by tokens
def filter_context_by_tokens(input, context_key, max_model_tokens, count_tokens):
    # Initialize cumulative token count
    cumulative_tokens = 0

    # Filtered list to store dictionaries
    filtered_context = []

    # Iterate through the list of dictionaries
    for item in input:
        # Get the context value based on the context_key
        context_value = item[context_key]
        
        # Calculate number of tokens for 'context' value
        token_count = count_tokens(context_value)
        
        # Cumulative sum of token counts
        cumulative_tokens += token_count
        
        # Check if cumulative tokens are still less than max_model_tokens
        if cumulative_tokens < max_model_tokens:
            filtered_context.append(item)
        else:
            break
    
    # Return
    return filtered_context

# Retrieval functions ----------------------------------------------

# Retrieve info function
def retrieve_context(message, k, vectorstore):
    # Similarity output
    similarity_output = vectorstore.similarity_search_with_score(message, k)
    # Create list of documents
    context_processed = []
    # Context processed
    for doc, score in similarity_output:
        metadata = doc.metadata
        if score < .5:
            continue
        context_processed.append({
            "date": metadata.get('date', ''),
            "title": metadata.get('title', ''),
            "context": doc.page_content,
            "url": metadata.get('url', ''),
            "score": score
        })
    # Return
    return context_processed

# Chat functions ----------------------------------------------

# Generate chat responses function
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Format context and exec_summary
def str_context(context, exec_summary, n = 6):
    c = '  \n'.join(exec_summary) + '  \n'
    for element in context[:n]:
        c += '  \n  \n'
        for k,v in element.items():
            c += f'{k}:  \t{v}  \n'
    return c

# Streamlit functions ----------------------------------------------

# Provide context details
def provide_context_details(context):
    # Get exec summary info
    num_documents_found = len(context)
    num_tokens_used = sum(count_tokens(item['context']) for item in context)
    current_date = dt.datetime.now().strftime("%Y/%m/%d, %H:%M:%S")
    # Exec summary
    exec_summary = (
        f"Fecha de ejecución: {current_date}",
        f"Documentos encontrados: {num_documents_found}",
        f"Número de tokens usados: {num_tokens_used}",
    )
    # Context fields
    context_fields = [{'Fecha': item['date'], 
                       'Título': item['title'], 
                       'URL': item['url'], 
                       'Puntuación': f"{int(100*item['score']+10)}%"} for item in context]
    # Append messages
    st.session_state.messages.append({"role": "assistant", "content": str_context(context_fields, exec_summary, 6)})

# Clear history function
def clear_chat_history():
    # Delete chat history
    st.session_state.conversation = []
    st.session_state.chat_history = []
    st.session_state.messages = []
    # Print message
    st.write("Historial de chat eliminado")
