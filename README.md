# Overview
This application leverages a Llama 3 powered Retrieval-Augmented Generation (RAG) pipeline to create an AI-powered purchase assistant. The application performs several steps to achieve its functionality:

1. **Data Indexing**: Reads a CSV file containing Amazon product data and indexes it in an in-memory vector database.
2. **User Query Analysis**: Analyzes user queries to identify products of interest.
Product Retrieval: Retrieves similar products from the indexed database based on the user's query.
3. **Tool Functions**: Provides functionalities such as identifying products and finding budget-friendly options.
4. **Chat Interface**: Uses a chat interface with function calling capabilities to interact with the user and provide recommendations.

# Getting started
Create a .env file with the following environment variables:

* GROQ_API_KEY=your-groq-api-key
* HF_API_TOKEN=your-hugging-face-api-token
* GROQ_LLM_MODEL=llama3-groq-70b-8192-tool-use-preview

# Detailed explanations
## Data Indexing:

The application reads a CSV file into a Pandas DataFrame.
It creates an in-memory document store and indexes the documents using the SentenceTransformersDocumentEmbedder.
This step ensures that the product data is transformed into embeddings that can be efficiently searched.

## User Query Analysis:

A product_identifier pipeline is created to analyze user queries.
The pipeline uses a PromptBuilder to generate a prompt and an OpenAIGenerator to process the prompt.
The template ensures the returned result is a Python list of products.

## Product Retrieval:

A RAG pipeline is created to retrieve similar products from the vector database.
The pipeline uses SentenceTransformersTextEmbedder to generate query embeddings and InMemoryEmbeddingRetriever to find similar documents.
A PromptBuilder formats the retrieved documents and a second OpenAIGenerator generates the final response.

## Tool Functions:

Two primary tool functions are defined:
product_identifier_func(query: str): Identifies products from the user's query and retrieves similar products.
find_budget_friendly_option(selected_product_details): Finds the most cost-friendly option from a list of products.

## Chat Interface:

The chat interface is built using the gradio library.
The chat template includes XML tags for tool calls, allowing the AI to call predefined functions to process user queries.
The chatbot_with_fc function handles the conversation, checking for tool calls and executing the corresponding functions.
The extract_tool_calls function parses tool call information from the AI's responses.
The chatbot_interface function integrates the chatbot logic with Gradio's interface elements.

# Key Components and Their Roles

## Gradio for User Interface:

gradio provides a web-based interface for interacting with the chatbot.
It allows users to input queries and view responses in a user-friendly manner.

## Haystack for Pipelines:

haystack is used to create and manage the various pipelines.
It provides components for embedding, retrieval, and generation tasks.

## Groq Llama 3 Language Model:

OpenAIGenerator and OpenAIChatGenerator utilize Groq's Llama models to generate responses and handle chat interactions.

## Environment Variables:

Environment variables are managed using dotenv to keep sensitive information secure and configurable.

## Error Handling and Data Parsing:

The use of literal_eval and JSON parsing ensures that responses are correctly interpreted and handled.

# Future Roadmap & Improvements

## Error Handling:

Enhancing error handling to cover more edge cases and provide informative error messages.
For example, logging errors for debugging and improve user feedback in case of failures.

## Scalability:

Scaling the document store and retrieval components to handle larger datasets and more simultaneous queries.
Using a more scalable document store like Elasticsearch for production environments.

## Performance Optimization:

Optimizing the embedding and retrieval steps to reduce latency.
Caching frequently retrieved results can improve response times.

## User Experience:

Improving the UI/UX of the Chat interface to make it more intuitive and visually appealing.
Adding features like conversational history, context retention, and more interactive elements.

## Security:

Ensuring that the application is secure, especially when handling user inputs and API keys.
Implementing rate limiting and input validation to prevent abuse and injection attacks.

# Conclusion
This RAG pipeline-based LLM agentic application combines various components to create a sophisticated AI-powered purchase assistant. By leveraging document embedding, retrieval, and generation, it provides users with relevant product recommendations and budget-friendly options. The integration with Gradio ensures a smooth and interactive user experience, while the use of environment variables and structured templates maintains flexibility and security.