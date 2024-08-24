import gradio as gr
import json
import os
import pandas as pd
import pprint
import re
from ast import literal_eval
from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret

# Load environment variables from .env file
load_dotenv()

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

# STEP 4: Create tools for the agent to use.
# Tool 1 - A wrapper function that uses both the query analyzer and RAG pipeline.


def product_identifier_func(query: str):
    product_understanding = product_identifier.run(
        {"prompt_builder": {"question": query}})
    # print("product_understanding: ", product_understanding)

    try:
        product_list = literal_eval(product_understanding["llm"]["replies"][0])
    except:
        return "Got an exception finding product list. No product found."

    results = {}

    for product in product_list:
        response = rag_pipe.run(
            {"embedder": {"text": product}, "prompt_builder": {"question": product}})
        try:
            results[product] = literal_eval(response["llm"]["replies"][0])
        except:
            results[product] = {}

    return results


# Test of first tool use
query = "I want crossbow and woodstock puzzle"
# execute function
print(product_identifier_func(query))

# Tool 2 - Tool which finds the cheapest option.


def find_budget_friendly_option(selected_product_details):
    budget_friendly_options = {}

    for category, items in selected_product_details.items():
        if isinstance(items, list):
            lowest_price_item = min(items, key=lambda x: x['price'])
        else:
            lowest_price_item = items

        budget_friendly_options[category] = lowest_price_item

    return budget_friendly_options


# Test second tool
print("running second tool find_budget_friendly_option: \n")
print(find_budget_friendly_option(product_identifier_func(query)))

# STEP 5: Add the Chat template with tool usage for our agent
chat_template = '''<|start_header_id|>system<|end_header_id|>

You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"name": <function-name>,"arguments": <args-dict>}
</tool_call>

Here are the available tools:
<tools>
    {
        "name": "product_identifier_func",
        "description": "To understand user interested products and its details",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to use in the search. Infer this from the user's message. It should be a question or a statement"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "find_budget_friendly_option",
        "description": "Get the most cost-friendly option. If selected_product_details has morethan one key this should return most cost-friendly options",
        "parameters": {
            "type": "object",
            "properties": {
                "selected_product_details": {
                    "type": "dict",
                    "description": "Input data is a dictionary where each key is a category name, and its value is either a single dictionary with 'price', 'name', and 'url' keys or a list of such dictionaries; example: {'category1': [{'price': 10.5, 'name': 'item1', 'url': 'http://example.com/item1'}, {'price': 8.99, 'name': 'item2', 'url': 'http://example.com/item2'}], 'category2': {'price': 15.0, 'name': 'item3', 'url': 'http://example.com/item3'}}"
                }
            },
            "required": ["selected_product_details"]
        }
    }
</tools><|eot_id|><|start_header_id|>user<|end_header_id|>

I need to buy a crossbow<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<tool_call>
{"id":"call_deok","name":"product_identifier_func","arguments":{"query":"I need to buy a crossbow"}}
</tool_call><|eot_id|><|start_header_id|>tool<|end_header_id|>

<tool_response>
{"id":"call_deok","result":{'crossbow': {'price': 237.68,'name': 'crossbow','url': 'https://www.amazon.com/crossbow/dp/B07KMVJJK7'}}}
</tool_response><|eot_id|><|start_header_id|>assistant<|end_header_id|>
'''

# # Testing agent
# messages = [
#     ChatMessage.from_system(
#         chat_template
#     ),
#     ChatMessage.from_user(
#         "I need to buy a crossbow for my child and Pokémon for myself."),
# ]

# chat_generator = OpenAIChatGenerator(api_key=Secret.from_env_var("GROQ_API_KEY"),
#                                      api_base_url="https://api.groq.com/openai/v1",
#                                      model=os.getenv('GROQ_LLM_MODEL'))
# response = chat_generator.run(messages=messages)
# pprint.pp(response)


def get_chat_generator():
    return OpenAIChatGenerator(api_key=Secret.from_env_var("GROQ_API_KEY"),
                               api_base_url="https://api.groq.com/openai/v1",
                               model=os.getenv('GROQ_LLM_MODEL'))

# tool calls are enclosed using the XML tag <tool_call>. Therefore, we have to build a mechanism to extract the tool_call object.


def extract_tool_calls(tool_calls_str):
    json_objects = re.findall(
        r'<tool_call>(.*?)</tool_call>', tool_calls_str, re.DOTALL)

    result_list = [json.loads(obj) for obj in json_objects]

    return result_list


available_functions = {
    "product_identifier_func": product_identifier_func,
    "find_budget_friendly_option": find_budget_friendly_option
}

# We can now directly access the agent’s response when it calls a tool. Now we need to get the tool call object and execute the function accordingly.

# STEP 6: Build a full fledged chat app
messages = [ChatMessage.from_system(chat_template)]
chat_generator = get_chat_generator()


def chatbot_with_fc(message, messages):
    messages.append(ChatMessage.from_user(message))
    response = chat_generator.run(messages=messages)

    while True:
        if response and "<tool_call>" in response["replies"][0].content:
            function_calls = extract_tool_calls(response["replies"][0].content)
            for function_call in function_calls:
                # Parse function calling information
                function_name = function_call["name"]
                function_args = function_call["arguments"]

                # Find the corresponding function and call it with the given arguments
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)

                # Append function response to the messages list using `ChatMessage.from_function`
                messages.append(ChatMessage.from_function(
                    content=json.dumps(function_response), name=function_name))
                response = chat_generator.run(messages=messages)

        # Regular Conversation
        else:
            messages.append(response["replies"][0])
            break
    return response["replies"][0].content


def chatbot_interface(user_input, state):
    response_content = chatbot_with_fc(user_input, state)
    return response_content, state

def get_chat_generator():
    return OpenAIChatGenerator(api_key=Secret.from_env_var("GROQ_API_KEY"),
                               api_base_url="https://api.groq.com/openai/v1",
                               model=os.getenv('GROQ_LLM_MODEL'))

# with gr.Blocks() as demo:
#     gr.Markdown("# AI Purchase Assistant")
#     gr.Markdown("Ask me about products you want to buy!")

#     state = gr.State(value=messages)

#     with gr.Row():
#         user_input = gr.Textbox(label="Your message:")
#         response_output = gr.Markdown(label="Response:")

#     user_input.submit(chatbot_interface, [user_input, state], [
#                       response_output, state])
#     gr.Button("Send").click(chatbot_interface, [
#         user_input, state], [response_output, state])


# demo.launch()
