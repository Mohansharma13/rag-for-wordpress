from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Logging configuration for debug information
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)  # Logger for logging events in the function

def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model, with Chain-of-Thought reasoning.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    # Log the question and selected model for debugging
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    # Initialize the language model with specific parameters
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    
    # Prompt template for generating alternative queries
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 
        different three versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    # Initialize the multi-query retriever
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    # Template with Chain-of-Thought prompting
    template = """
        You are an AI language model for . Answer the question using the following context and your own knowledge.
        
        Before you give the final answer, think through the problem step-by-step to make sure the reasoning is clear:
        Context: {context}
        Question: {question}
        
        Think step-by-step to arrive at the answer. If the context does not provide sufficient information, use your own knowledge.
        However, if you still don't know the answer, simply say that you don't know.
    """

    # Create the chat prompt with CoT
    prompt = ChatPromptTemplate.from_template(template)

    # Define the process chain with CoT
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the chain with the question and get the response
    response = chain.invoke(question)
    logger.info("Question processed and response generated")  # Log success

    return response
