# Credit Buddy: Credit Card Recommendation System

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Technologies Used](#models--technologies-used)

## Introduction
Credit Buddy is an interactive AI-powered recommendation system that helps users find the best credit cards tailored to their financial profile and preferences. Leveraging Retrieval-Augmented Generation (RAG), semantic search, and large language models (LLMs), Credit Buddy analyzes user inputs and retrieves relevant information from a curated knowledge base of credit card offerings in India.

The core components of Credit Buddy include:
- A knowledge base built from top credit card documents
- Embedding and semantic indexing of the knowledge base
- Pinecone, a cloud vector database for similarity search
- A query embedding and search module to find the most relevant information
- An LLM for answer refinement and personalized recommendations

## System Overview and Methodology

### 1. Knowledge Base Creation
Text content is extracted from credit card documents and split into manageable chunks for efficient processing.

### 2. Embedding and Semantic Indexing
Text chunks are converted into vector representations using a state-of-the-art embedding model. These vectors are stored in Pinecone for fast similarity search.

### 3. User Query Processing
User queries are embedded and matched against the indexed knowledge base to find the most relevant information.

### 4. Answer Retrieval and Recommendation
Relevant chunks are retrieved and passed to a large language model, which generates a concise, user-friendly recommendation based on the user's profile and preferences.

## Features
- Personalized credit card recommendations
- Natural language understanding and processing
- Semantic search for relevant information retrieval
- Integration with a comprehensive credit card knowledge base
- Generative responses using a large language model
- User-friendly web interface

## Installation
To run Credit Buddy locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/credit-buddy.git
   ```

2. Install the required dependencies:
   ```
   # Create a virtual environment
   python -m venv creditbuddyEnv
   # Activate the environment and install dependencies
   pip install -r requirements.txt
   ```

3. Set up the necessary environment variables:
   - `PINECONE_API_KEY` and `PINECONE_ENV`: For Pinecone vector database access
   - `HUGGINGFACE_API_KEY`: For HuggingFace Inference Endpoint access

4. Run the application:
   ```
   python main.py
   ```

5. Access the web interface through the provided URL (default: http://localhost:8080).

## Usage
Interact with Credit Buddy by answering a few onboarding questions about your income, spending habits, preferred benefits, existing cards, and credit score. Then, ask for credit card recommendations or information, such as:
- "Which credit card is best for travel rewards?"
- "Suggest a card with good cashback for shopping."
- "What are the eligibility criteria for premium cards?"

Credit Buddy will analyze your profile and provide tailored recommendations with clear comparisons and feature highlights.

## Models & Technologies Used
- **Embedding Model:** [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5) (HuggingFace)
- **LLM:** [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) (HuggingFace Inference Endpoint)
- **Vector Database:** [Pinecone](https://www.pinecone.io/)
- **Frameworks/Libraries:** Flask, LangChain, dotenv

---

Credit Buddy makes finding the right credit card easy, fast, and personalized. For any questions or contributions, please open an issue or pull request!
