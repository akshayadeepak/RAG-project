"""Question and Answering system based on Singapore wikipedia page"""

import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline

# scrape wikipedia page content
url = 'https://en.wikipedia.org/wiki/Singapore'
response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

paragraphs = soup.find_all('p')
page_content = [para.text for para in paragraphs]

# use headers as metadata (maybe)
# title = soup.find('h1', id="firstHeading").text
#
# headers = soup.find_all(['h2', 'h3'])
# genres = [header.get_text(strip=True) for header in headers]

CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "default"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

# check if collection already exists
# so that you don't have to keep changing the collection name each time
try:
    collection = client.get_collection(COLLECTION_NAME)
except:
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

# add documents only if the collection was newly created
if not collection.count():
    collection.add(
         documents=page_content,
         ids=[f"id{i}" for i in range(len(page_content))],
    )

# keep querying until user inputs "quit"
while True:
    question = input("Please enter your question: ")

    if question == "quit":
        break

    query_results = collection.query(
         query_texts=[question],
         n_results=1,
    )

    # extract most relevant document
    retrieved_text = query_results["documents"][0][0]

    # use hugging face model to generate answer
    model = pipeline("question-answering")

    print("Text: ", retrieved_text)

    answer = model({
        'question': question,
        'context': retrieved_text
    })

    print("Answer: ", answer['answer'])

# question -> relevant para -> use as context to generate answer
