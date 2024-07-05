# langchain version 0.2, 07/03/2024
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
import chromadb
from chromadb import Documents, EmbeddingFunction

import os
from transformers import AutoModel, AutoTokenizer
from typing import List

# config
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DEVICE = "cuda"
CONSTRUCT_CHROMA = False
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
# class TreeEmbedding:
#     def __init__(self, checkpoint):
#         self.model = GNNModel().load_state_dict(torch.load(checkpoint)).eval()
#         self.encode = 
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return [self.model.encode(t).tolist() for t in texts]
    
#     def embed_query(self, query: str) -> List[float]:
#         return self.model.encode([query])
        

def create_seed_data(model, num_of_data, output_dir):
    """
    This function takes in a model and creates a seed dataset. 

    Args: 
        model (ChatOpenAI model): This is an initialized model that we will use to generate the data.
                                    Assumed to be gpt-4o.
        num_of_data (int): This is the amount of synthetic data to generate for the seed dataset
        output_dir (string): The path to the directory of where the dataset should be written to. 

    Return: 
        None: This just writes to file, so doesn't have an official return
    """
    print("temp")


if __name__ == "__main__":

    # first we read in our data and preprocess and store it in a chroma vector db based on the appropriate embedding method
    # setup for codet5p embedding

    data_dir = "./dataset/test_code"
    checkpoint = "Salesforce/codet5p-110m-embedding"

    # this is for the codet5p embedding case
    embedding_function_chroma_codet = ChromaCodet5pEmbedding(checkpoint)

    # if construct db is true
    if CONSTRUCT_CHROMA:
        raw_docs = DirectoryLoader(data_dir, glob='*.py', loader_cls=TextLoader).load()
        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
        docs = text_splitter.split_documents(raw_docs)

        persistent_client = chromadb.PersistentClient() # this is using default settings, so imports will also need defaults
        collection = persistent_client.get_or_create_collection("seed_code", embedding_function=embedding_function_chroma_codet)
        for i, entry in enumerate(docs):
            collection.add(ids=f"{i}", embeddings = embedding_function_chroma_codet(entry.page_content), metadatas=entry.metadata, documents=entry.page_content)

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
    