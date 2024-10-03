# langchain version 0.2, 07/03/2024
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
import chromadb
from chromadb import Documents, EmbeddingFunction

from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

import os
from transformers import AutoModel, AutoTokenizer
from typing import List

from tree_encoder_model import GNNModel
from code_preprocessing import preprocess_tree_query
import torch

from tree_model_trainer import EMBED_DIM, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NODE_CLASSES, COMPRESSED_CLASSES, COMPRESSED_GRAPH_FEATURE, GRAPH_FEATURE

# config
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DEVICE = "cuda"
CONSTRUCT_CHROMA = False
CONSTRUCT_CHROMA_TREE = False
CHROMA_TREE = False
CONSTRUCT_CHROMA_NOMIC = True
NOMIC_CHECKPOINT = "nomic-ai/nomic-embed-text-v1.5"
CHECKPOINT = "Salesforce/codet5p-110m-embedding"
TREE_CHECKPOINT = "./models/tree_ae/tree_ae.pth"
URL = "http://localhost:8000/code2fn/fn-given-filepaths"
    
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

class ChromaNomicEmbedding(EmbeddingFunction):
    def __init__(self, checkpoint, matryoshka_dim):
        self.model =SentenceTransformer(checkpoint, trust_remote_code=True)
        self.matryoshka_dim = matryoshka_dim
    
    def __call__(self, texts: Documents) -> chromadb.Embeddings:
        """This is a little hack-y right now, need to read more documentation
            to see how to get this working more cleanly."""
        #print(len(texts))
        if len(texts) == 1:
            texts = texts[0]
        #print(len(tokens_vec))
        # This has a lot of extra steps due to the Matryoshka Dimensional Fixing
        matryoshka_dim = self.matryoshka_dim
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        #print(embeddings.shape)
        #embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[0],))
        embeddings = embeddings[:matryoshka_dim]
        embeddings = F.normalize(embeddings, p=2, dim=0).tolist()
        #print(len(embeddings))
        if len(embeddings) == 1:
            return embeddings[0]
        else:
            return embeddings

# We need to define an embedding function in langchain style to use it in Chroma vector db
# this is also a bit janky since I didn't define the tree encoding function as a class
class ChromaTreeEmbedding(EmbeddingFunction):
    def __init__(self, model_checkpoint, tree_checkpoint, url):
        self.model = GNNModel(EMBED_DIM, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NODE_CLASSES, COMPRESSED_CLASSES, COMPRESSED_GRAPH_FEATURE, GRAPH_FEATURE)
        self.url = url
        self.model_checkpoint = model_checkpoint
        self.tree_checkpoint = tree_checkpoint

    def __call__(self, texts: Documents) -> chromadb.Embeddings:
        """This is a little hack-y right now, need to read more documentation
            to see how to get this working more cleanly."""
    
        embeddings = []
        em_model = self.model
        em_model.load_state_dict(torch.load(self.tree_checkpoint))
        em_model.eval()
        
        # regular calls send a string query sends the string in a list
        if isinstance(texts, list):
            if len(texts) == 1:
                texts = texts[0]
            else:
                print("error, large list input")

        encoded_graph = preprocess_tree_query(texts, self.url, self.model_checkpoint)
        if encoded_graph is not None:
            if encoded_graph[0].pe.shape[1] != 12:
                embeddings.append(torch.zeros(GRAPH_FEATURE).tolist())
            else:
                embeddings.append(em_model.encoder(encoded_graph[0].x1, encoded_graph[0].x2, encoded_graph[0].pe, encoded_graph[0].edge_index, encoded_graph[0].batch).tolist())

            # they have a type checker that requires since inputs to output a single vector but a collection of inputs output a list
            # of vectors
            if len(embeddings) == 1:
                return embeddings[0]
            else:
                return embeddings
        else:
            embeddings.append(torch.zeros(GRAPH_FEATURE).tolist())
            if len(embeddings) == 1:
                return embeddings[0]
            else:
                return embeddings

if __name__ == "__main__":

    # first we read in our data and preprocess and store it in a chroma vector db based on the appropriate embedding method
    # setup for codet5p embedding

    data_dir = "./dataset/new_test_code"
    checkpoint = CHECKPOINT
    tree_checkpoint = TREE_CHECKPOINT
    url = URL

    # if construct db is true
    if CONSTRUCT_CHROMA:
        embedding_function_chroma_codet = ChromaCodet5pEmbedding(checkpoint)
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
        embedding_function_chroma_tree = ChromaTreeEmbedding(checkpoint, tree_checkpoint, url)
        # this is for testing the tree embedding function wrapping in chroma
        # no need to chunk the inpur for this case
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
    elif CHROMA_TREE:
        # make a query of a previous code file
        raw_docs = DirectoryLoader(data_dir, glob='*.py', loader_cls=TextLoader).load()

        embedding_function_chroma_tree = ChromaTreeEmbedding(checkpoint, tree_checkpoint, url)
        persistent_client = chromadb.PersistentClient() # default settings
        # this gets the collection since it's already present
        collection = persistent_client.get_collection("seed_code_tree", embedding_function=embedding_function_chroma_tree)
        print("There are", collection.count(), "in the collection")
        results = collection.query(
            query_texts=[raw_docs[0].page_content], # Chroma will embed this for you
            n_results=2 # how many results to return
        )
        print(results['distances']) # this is a dict of 'ids', 'distances', 'metadatas', 'embeddings', 'documents', 'uris', 'data', 'included'
        #print(results["documents"])
    elif CONSTRUCT_CHROMA_NOMIC:
        data_dir = "./dataset/new_test_code"
        matryoshka_dim = 128

        embedding_function_chroma_nomic = ChromaNomicEmbedding(NOMIC_CHECKPOINT, matryoshka_dim)
        raw_docs = DirectoryLoader(data_dir, glob='*.py', loader_cls=TextLoader).load()

        persistent_client = chromadb.PersistentClient() # this is using default settings, so imports will also need defaults
        collection = persistent_client.get_or_create_collection("seed_code_nomic", embedding_function=embedding_function_chroma_nomic)
        
        for i, entry in enumerate(raw_docs):
            collection.add(ids=f"{i}", embeddings = embedding_function_chroma_nomic(entry.page_content), metadatas=entry.metadata, documents=entry.page_content)
            print(f"{i} of {len(raw_docs)} added to db")

        results = collection.query(
            query_texts=["def simulate_seir_model(N, I0, E0, R0, beta, gamma, sigma, days):"], # Chroma will embed this for you
            n_results=2 # how many results to return
        )

        print(results)
    else:
        embedding_function_chroma_codet = ChromaCodet5pEmbedding(checkpoint)
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
    