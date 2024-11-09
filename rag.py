import streamlit as st
import logging
# import ollama  # Uncomment if using a library like Ollama local for models
from typing import List, Tuple, Dict, Any, Optional

import streamlit.components.v1 as components
# import user-defined modules for the RAG application
from chat_model import process_question
from vectordb import create_vector_db_from_text, delete_vector_db
from web_to_text import extract_all_pages, is_valid_url


# On Windows systems, switch to an alternative SQLite package for compatibility or comment the code below
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')




# Streamlit page configuration
st.set_page_config(
    page_title="Insight Matrix RAG",          # Sets page title
    page_icon="💼",                           # Sets favicon
    layout="wide",                            # Sets layout to wide-screen
    initial_sidebar_state="collapsed",        # Hides sidebar on page load
)

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)  # Logger for error and info messages

@st.cache_resource(show_spinner=True)
def extract_model_names():
    """
    Function to extract model names. For now, it returns a static model name
    until the model list extraction from Ollama (or other source) is configured.
    """
    # Placeholder for model extraction, currently returns "gemma_model"
    model_names = "gemma_model"
    return model_names

def main() -> None:
    """
    Main function to run the Streamlit application.
    
    Sets up the user interface, handles file uploads, and processes user queries.
    """
    # Display application title
    st.subheader("IntelliQuest 📈", divider="gray", anchor=False)

    # Get available model names
    available_models = extract_model_names()  
    
    # Create layout with two columns
    col1, col2 = st.columns([2, 1.5])
    
    # Initialize session state variables if not already present
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    # Dropdown to select an available model
    if available_models:
        selected_model = col1.selectbox(
            "Pick a model available  ↓", available_models
        )
   
    # Set default values for session state if not already initialized

    st.session_state.placeholder = "https://example.com"  # Default placeholder URL
    # Input box to receive WordPress website URL with HTML for spacing
    file_upload = col2.text_input(
        "Enter WordPress website URL: Example : https://divimundo.com",  # The label for the input box
        placeholder=st.session_state.placeholder,  # Set placeholder text to URL
    )
    
        
    file_valid = is_valid_url(file_upload)  # Validate the entered URL
    
    # Warn user if the entered URL is invalid
    if not file_valid:
        st.warning("Please Upload correct link")
    
    # Process URL and extract web pages if the URL is valid
    if file_upload and file_valid:
        
        # Save file URL with API endpoint in session state
        st.session_state["file_upload"] = file_upload + "/wp-json/wp/v2/posts"
        
        # Create vector database if not already created
        if st.session_state["vector_db"] is None:
            pages = extract_all_pages(file_upload + "/wp-json/wp/v2/posts")
            st.session_state["vector_db"] = create_vector_db_from_text(pages)
            col2.success('Web pages have been loaded!', icon="✅")
        
        # Extract pages as `pdf_pages` from the uploaded link for display or analysis
        pdf_pages = extract_all_pages(file_upload + "/wp-json/wp/v2/posts")
        st.session_state["pdf_pages"] = pdf_pages
        
        # Embed the entered URL in an iframe for display
        with col2:
            # components.iframe(file_upload, height=520)
            if file_upload and file_valid and "file_upload" in st.session_state:
                iframe_code = f'''
                <iframe src="{file_upload}" width="100%" height="520" style="border:none;" scrolling="yes"></iframe>
                '''
                st.markdown(iframe_code, unsafe_allow_html=True)

    # Button to delete the vector database collection
    delete_collection = col2.button("⚠️ Delete stored database collection", type="secondary")
    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    # Chat input and message display container
    with col1:
        message_container = st.container(height=500, border=True)

        # Display each chat message stored in session state
        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        # Input box for user prompt
        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                # Add user message to session state and display it
                st.session_state["messages"].append({"role": "user", "content": prompt})
                message_container.chat_message("user", avatar="😎").markdown(prompt)

                # Generate and display assistant's response
                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a correct WordPress website URL first.")

                # Store assistant's response in session state
                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")

        else:
            # Display warning if no database has been created for chat
            if st.session_state["vector_db"] is None:
                st.warning("Upload a correct web page URL to begin chat...")


# Entry point of the application
if __name__ == "__main__":
    main()
