# langchain version 0.2, 07/03/2024
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
import chromadb
from chromadb import Documents, EmbeddingFunction

import os
from transformers import AutoModel, AutoTokenizer
from typing import List

from tree_encoder_model import GNNModel
from code_preprocessing import preprocess_tree_query
import torch

# config
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DEVICE = "cuda"
CONSTRUCT_CHROMA = False
CONSTRUCT_CHROMA_TREE = True
CHECKPOINT = "Salesforce/codet5p-110m-embedding"
    
class ChromaCodet5pEmbedding(EmbeddingFunction):
    def __init__(self, checkpoint):
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
        self.tokenize = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    
    def __call__(self, texts: Documents) -> chromadb.Embeddings:
        """This is a little hack-y right now, need to read more documentation
            to see how to get this working more cleanly."""
        #print(len(texts))
        if len(texts) == 1:
            texts = texts[0]
        tokens_vec = self.tokenize.encode(texts, return_tensors="pt")
        #print(len(tokens_vec))
        embeddings = self.model(tokens_vec).tolist()
        #print(len(embeddings))
        if len(embeddings) == 1:
            return embeddings[0]
        else:
            return embeddings

# We need to define an embedding function in langchain style to use it in Chroma vector db
# this is also a bit janky since I didn't define the tree encoding function as a class
class ChromaTreeEmbedding(EmbeddingFunction):
    def __init__(self, model_checkpoint, tree_checkpoint, url):
        self.model = GNNModel().load_state_dict(torch.load(tree_checkpoint)).eval()
        self.url = url
        self.model_checkpoint = model_checkpoint

    def __call__(self, texts: Documents) -> chromadb.Embeddings:
        """This is a little hack-y right now, need to read more documentation
            to see how to get this working more cleanly."""
    
        embeddings = []
        for text in texts:
            encoded_graph = preprocess_tree_query(text.text, self.url, self.model_checkpoint)
            embeddings.append(self.model(encoded_graph).tolist())
    
        if len(embeddings) == 1:
            return embeddings[0]
        else:
            return embeddings

if __name__ == "__main__":

    # first we read in our data and preprocess and store it in a chroma vector db based on the appropriate embedding method
    # setup for codet5p embedding

    data_dir = "./dataset/new_test_code"
    checkpoint = "Salesforce/codet5p-110m-embedding"
    tree_checkpoint = ""
    url = "http://localhost:8000/code2fn/fn-given-filepaths"

    # this is for the codet5p embedding case
    embedding_function_chroma_codet = ChromaCodet5pEmbedding(checkpoint)
    embedding_function_chroma_tree = ChromaTreeEmbedding(checkpoint, tree_checkpoint, url)

    # if construct db is true
    if CONSTRUCT_CHROMA:
        raw_docs = DirectoryLoader(data_dir, glob='*.py', loader_cls=TextLoader).load()
        text_splitter = CharacterTextSplitter(chunk_size=456, chunk_overlap=0)
        docs = text_splitter.split_documents(raw_docs)

        persistent_client = chromadb.PersistentClient() # this is using default settings, so imports will also need defaults
        collection = persistent_client.get_or_create_collection("seed_code", embedding_function=embedding_function_chroma_codet)
        for i, entry in enumerate(docs):
            collection.add(ids=f"{i}", embeddings = embedding_function_chroma_codet(entry.page_content), metadatas=entry.metadata, documents=entry.page_content)
            print(f"{i} of {len(docs)} added to db")

        results = collection.query(
            query_texts=["def simulate_seir_model(N, I0, E0, R0, beta, gamma, sigma, days):"], # Chroma will embed this for you
            n_results=2 # how many results to return
        )
        print(results)

    elif CONSTRUCT_CHROMA_TREE:
        # this is for testing the tree embedding function wrapping in chroma
        raw_docs = DirectoryLoader(data_dir, glob='*.py', loader_cls=TextLoader).load()
        persistent_client = chromadb.PersistentClient() # this is using default settings, so imports will also need defaults
        collection = persistent_client.get_or_create_collection("seed_code_tree", embedding_function=embedding_function_chroma_tree)

        for i, entry in enumerate(raw_docs):
            collection.add(ids=f"{i}", embeddings = embedding_function_chroma_tree(entry.page_content), metadatas=entry.metadata, documents=entry.page_content)
            print(f"{i} of {len(raw_docs)} added to db")

        results = collection.query(
            query_texts=["def simulate_seir_model(N, I0, E0, R0, beta, gamma, sigma, days):"], # Chroma will embed this for you
            n_results=2 # how many results to return
        )
        print(results)

    else:
        persistent_client = chromadb.PersistentClient() # default settings
        # this gets the collection since it's already present
        collection = persistent_client.get_collection("seed_code", embedding_function=embedding_function_chroma_codet)
        print("There are", collection.count(), "in the collection")
        results = collection.query(
            query_texts=["def simulate_seir_model(N, I0, E0, R0, beta, gamma, sigma, days):"], # Chroma will embed this for you
            n_results=2 # how many results to return
        )
        print(results['distances']) # this is a dict of 'ids', 'distances', 'metadatas', 'embeddings', 'documents', 'uris', 'data', 'included'
        print(results["documents"])
    

    # the tree encoding is more complicated, but the following is for that. 
    