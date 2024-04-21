# Medical Drug and News Search Tool

## Overview

The Medical Drug Search Tool is a cutting-edge application designed to streamline the process of querying medical information regarding pharmaceutical drugs. It integrates advanced natural language processing (NLP) to allow users to input queries in natural language and quickly receive concise, summarized information pulled from a vast database of medical literature.

This prototype demonstrates the core functionalities of querying, data processing, and information summarization, all tailored to enhance the productivity of medical professionals and researchers.

## Features

- **Natural Language Queries**: Users can type queries in natural language to find information about medical drugs, their competitors, recent studies, and regulatory statuses.
- **Real-Time Summarization**: Leverages a Large Language Model (LLM) to summarize relevant medical texts from articles, providing users with accurate and digestible information.
- **Interactive User Interface**: A clean, responsive web interface that makes it easy for users to interact with the tool and view results.
- **Advanced API Integration**: Utilizes the PubMed API to fetch the latest and most relevant medical articles and information.
- **Scalable Architecture**: Built on Flask, offering robust and scalable server-side logic capable of handling numerous requests seamlessly.

## Technology Stack

- **Frontend**: HTML, CSS, JavaScript (with AJAX for asynchronous requests)
- **Backend**: Flask (Python)
- **APIs**: PubMed API, OpenAI API for GPT-3.5 Turbo
- **Natural Language Processing**: OpenAI’s GPT-3.5 Turbo
- **Development Tools**: Git for version control, Visual Studio Code as the IDE

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Pip (Python package installer)
- Access to OpenAI and PubMed APIs

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/medical-drug-search-tool.git
   cd medical-drug-search-tool
   ```

2. **Install required Python libraries**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   - Create a `.env` file in the project root.
   - Add your API keys:
     ```plaintext
     OPENAI_API_KEY='your_openai_api_key_here'
     NCBI_API_KEY='your_pubmed_api_key_here'
     ```

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open a web browser and navigate to `http://localhost:8000` to start using the application.
![Example Image](/Picture1.png)

## Future Development

Future enhancements will focus on integrating additional APIs, improving the summarization algorithms, and expanding the tool's language capabilities to support global usage. Plans include the development of a more interactive user interface and the implementation of a chatbot for conversational interactions.
