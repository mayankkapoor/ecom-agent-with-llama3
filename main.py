import pandas as pd
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret

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
print("Documents indexed.\n")

# Now you have indexed your documents in the in-memory vector database
# Let's create the first Product Identifier agent/tool which gets the product names from the user's query

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

product_identifier.add_component("prompt_builder",
                                 PromptBuilder(template=template))
product_identifier.add_component(
    "llm",
    OpenAIGenerator(api_key=Secret.from_env_var("GROQ_API_KEY"),
                    api_base_url="https://api.groq.com/openai/v1",
                    model="llama3-groq-70b-8192-tool-use-preview",
                    generation_kwargs={"max_tokens": 512}))

product_identifier.connect("prompt_builder", "llm")
print("Product identifier tool added to pipeline.\n")
