import pandas as pd
from ast import literal_eval
from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
import os

# STEP 1: Create an in-memory vector database that stores all your data

# Read the CSV file into a dataframe with pandas
df = pd.read_csv("./amazon_product_sample.csv")

# Create an in-memory document store
document_store = InMemoryDocumentStore()

# Create documents from the dataframe
documents = [
    Document(content=item.product_name,
             meta={
                 "id": item.uniq_id,
                 "price": item.selling_price,
                 "url": item.product_url
             }) for item in df.itertuples()
]

# Create the indexing pipeline
indexing_pipeline = Pipeline()

# Add components to the pipeline
# Initialize the embedding model & add it to pipeline
indexing_pipeline.add_component(instance=SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"),
    name="doc_embedder")
print("Embedding model added to pipeline.\n")

# Initialize the Doc Writer & add it to pipeline
indexing_pipeline.add_component(
    instance=DocumentWriter(document_store=document_store), name="doc_writer")

# Connect pipeline components & create the pipeline
indexing_pipeline.connect("doc_embedder.documents", "doc_writer.documents")

# Run the pipeline with the documents
indexing_pipeline.run({"doc_embedder": {"documents": documents}})
print("STEP 1 Data indexed & stored.\n")

# Now we have indexed our documents in the in-memory vector database

# STEP 2: Create user query analyser & product identifier that returns a python list of products in the user's query

template = """
Understand the user query and list of products the user is interested in and return product names as list.
You should always return a Python list. Do not return any explanation.

Examples:
Question: I am interested in camping boots, charcoal and disposable rain jacket.
Answer: ["camping_boots","charcoal","disposable_rain_jacket"]

Question: Need a laptop, wireless mouse, and noise-cancelling headphones for work.
Answer: ["laptop","wireless_mouse","noise_cancelling_headphones"]

Question: {{ question }}
Answer:
"""

product_identifier = Pipeline()

product_identifier.add_component(
    "prompt_builder", PromptBuilder(template=template))

product_identifier.add_component(
    "llm",
    OpenAIGenerator(api_key=Secret.from_env_var("GROQ_API_KEY"),
                    api_base_url="https://api.groq.com/openai/v1",
                    model=os.getenv('GROQ_LLM_MODEL'),
                    generation_kwargs={"max_tokens": 512})
)

product_identifier.connect("prompt_builder", "llm")
print("STEP 2 Product identifier tool added to pipeline.\n")

# STEP 3: Create a RAG pipeline that takes as input a python list of products
# and returns similar matching products from your vector database

# The template provided below for RAG aims to format the retrieved product
# information in a structured manner. This template instructs the model to
# format the output as a Python dictionary or a list of dictionaries with product details.

template = """
Return product name, price, and url as a python dictionary. 
You should always return a Python dictionary with keys price, name and url for single product.
You should always return a Python list of dictionaries with keys price, name and url for multiple products.
Do not return any explanation.

Legitimate Response Schema:
{"price": "float", "name": "string", "url": "string"}
Legitimate Response Schema for multiple products:
[{"price": "float", "name": "string", "url": "string"},{"price": "float", "name": "string", "url": "string"}]

Context:
{% for document in documents %}
    product_price: {{ document.meta['price'] }}
    product_url: {{ document.meta['url'] }}
    product_id: {{ document.meta['id'] }}
    product_name: {{ document.content }}
{% endfor %}
Question: {{ question }}
Answer:
"""

# Initializes a new pipeline for RAG.
rag_pipe = Pipeline()

# SentenceTransformersTextEmbedder: This component is responsible for generating embeddings for the text input using the specified model.
rag_pipe.add_component("embedder", SentenceTransformersTextEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"))

# InMemoryEmbeddingRetriever: This component retrieves the top k similar documents based on the embedding generated. It uses the in-memory document store created earlier.
rag_pipe.add_component("retriever", InMemoryEmbeddingRetriever(
    document_store=document_store, top_k=5))

# PromptBuilder: This component formats the context and the question into a prompt that will be sent to the language model.
rag_pipe.add_component("prompt_builder", PromptBuilder(template=template))

# This component generates the final response using the LLM
rag_pipe.add_component("llm",
                       OpenAIGenerator(api_key=Secret.from_env_var("GROQ_API_KEY"),
                                       api_base_url="https://api.groq.com/openai/v1",
                                       model=os.getenv('GROQ_LLM_MODEL'),
                                       generation_kwargs={"max_tokens": 512})
                       )

# Connects the embedding output of the SentenceTransformersTextEmbedder to the query embedding input of the InMemoryEmbeddingRetriever.
rag_pipe.connect("embedder.embedding", "retriever.query_embedding")

# Connects the output of the retriever (the documents retrieved) to the input of the PromptBuilder.
rag_pipe.connect("retriever", "prompt_builder.documents")

# Connects the output of the PromptBuilder (the formatted prompt) to the input of the LLM.
rag_pipe.connect("prompt_builder", "llm")

print("STEP 3 RAG pipeline for retrieving similar products created")

# STEP 4: Create a wrapper function that uses both the query analyzer and RAG pipeline.
def product_identifier_func(query: str):
    product_understanding = product_identifier.run({"prompt_builder": {"question": query}})
    # print("product_understanding: ", product_understanding)

    try:
        product_list = literal_eval(product_understanding["llm"]["replies"][0])
    except:
        return "Got an exception finding product list. No product found."

    results = {}

    for product in product_list:
        response = rag_pipe.run({"embedder": {"text": product}, "prompt_builder": {"question": product}})
        try:
            results[product] = literal_eval(response["llm"]["replies"][0])
        except:
            results[product] = {}
    
    return results

# Test
query = "I want crossbow and woodstock puzzle"
#execute function
print(product_identifier_func(query))
