# RAG App with Chain of Thought (CoT) and Multi-Query Retrieval For Wordpress website Blog

This project is a Retrieval-Augmented Generation (RAG) application that enhances response accuracy and depth by implementing a Chain of Thought (CoT) approach. By generating multiple alternative queries, the system retrieves a more comprehensive set of documents to answer complex user questions with improved relevance and context.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [How It Works](#how-it-works)
6. [Conclusion](#conclusion)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview

This RAG app uses LangChain, a powerful framework for AI applications, to create a system capable of answering user questions with augmented responses. By employing a Chain of Thought (CoT) approach, the app generates multiple interpretations of each question to retrieve related documents from a vector database. The end result is a well-rounded and contextually informed response that mimics human-like reasoning.

## Features

- **Chain of Thought (CoT) Multi-Query Generation**: Generates multiple alternative versions of the user's question to retrieve a broader set of relevant documents.
- **Retrieval-Augmented Generation (RAG)**: Provides accurate answers by retrieving related documents and combining them with the generative model's response.
- **Vector Database Integration**: Uses Chroma as a vector store to manage document embeddings.
- **Web Scraping**: Extracts and processes data from web pages for use in the vector database.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Mohansharma13/rag-for-wordpress.git
    ```
2. Navigate to the project directory:
    ```bash
    cd rag-app-with-cot
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Set up your environment variables:
    - Create a `.env` file in the root directory.
    - Add your `GOOGLE_API_KEY` and any other required API keys.

## Usage

1. **Run the Application**: Launch the app using Streamlit:
    ```bash
    streamlit run app.py
    ```
2. **Interact with the RAG System**: Enter a question, and the system will generate multiple interpretations to retrieve the most relevant documents and respond accurately.

## How It Works

1. **Data Retrieval and Vectorization**:
   - Data is retrieved from a specified URL using a custom web scraper.
   - After retrieving the raw content, it's cleaned of HTML tags and irrelevant shortcodes.
   - The cleaned text is converted into document embeddings using `sentence-transformers/all-MiniLM-L6-v2`, then stored in a Chroma vector database.

2. **Chain of Thought (CoT) Multi-Query Generation**:
   - For each user query, the app generates multiple alternative queries through a prompt template in LangChain. This multi-query retriever enhances accuracy by capturing various angles of the initial question.
   
3. **Multi-Query Retrieval**:
   - The generated queries are used to retrieve documents from the vector database.
   - By combining the results of these queries, the system assembles a richer context for answering the question.

4. **Response Generation**:
   - The retrieved documents are passed to a generative AI model (e.g., `ChatGoogleGenerativeAI`) to produce a comprehensive response.
   - If the context is insufficient, the model relies on its training knowledge to generate a complete answer.

## Conclusion

The CoT-based multi-query strategy provides a smarter way of handling complex and open-ended questions, allowing the app to deliver responses that are both accurate and contextually enriched. This approach demonstrates the power of retrieval-augmented generation in real-world AI applications.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with any enhancements or bug fixes.

## License

This project is licensed under the MIT License.
