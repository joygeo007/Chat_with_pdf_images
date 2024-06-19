

## Link to Demo
https://chatwithpdfimages.streamlit.app/


# PDF Query Bot

PDF Query Bot is an intelligent chatbot application designed to help users query and extract information from PDF documents. The bot leverages OpenAI's GPT-3.5 model and embedding techniques to provide accurate answers based on the content of the PDFs.

## Features

- **PDF Text Extraction**: Extracts text and images from PDF files.
- **Text Chunking**: Splits the extracted text into smaller snippets for better processing.
- **Embeddings Creation**: Generates embeddings for the text snippets using OpenAI's embedding model.
- **Question Answering**: Matches user queries with relevant snippets from the PDFs and generates a response using GPT-3.5.
- **Streamlit Interface**: Provides an interactive web interface for users to interact with the bot and visualize the results.

## Requirements

- Python 3.7+
- Streamlit
- pypdf
- langchain
- openai
- numpy
- fitz (PyMuPDF)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/pdf-query-bot.git
    cd pdf-query-bot
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your OpenAI API key. You can either:
   - Replace the placeholder in the code with your API key, or
   - Store the API key in an environment variable.

4. Run the application:

    ```bash
    streamlit run app.py
    ```

## Usage

1. **Upload PDF Files**: Place your PDF files in the designated folder.
2. **Load PDFs**: The bot extracts text and images from the PDFs.
3. **Create Embeddings**: The text is split into snippets and embeddings are created.
4. **Query the Bot**: Use the Streamlit interface to ask questions. The bot will provide answers based on the content of the PDFs and display relevant images if available.

## File Structure

- **app.py**: Main Streamlit application.
- **app_utils.py**: Utility functions for text extraction, embeddings creation, and question answering.
- **requirements.txt**: List of required packages.



## Contributing

Contributions are welcome! Please create an issue or submit a pull request for any improvements or bug fixes.

