# langchain version 0.2, 07/03/2024
# for the agent
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# for the tool construction
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool

# for rag specifically, chroma
import chromadb
from chromadb import Documents, EmbeddingFunction
from vector_db import ChromaCodet5pEmbedding, CHECKPOINT

# generic
import os

"""
The goal of this agent is the following:
    1) Creates synthetic code data for a specific narrow domain 
    2) Interact with a vector db using embedding tools built we have to determine similarity
    3) Enirch an existing dataset based on conditions of the similarity of the new generated data
"""
# config
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DEVICE = "cuda"
# both agents below are gpt-4o based
RAG_AGENT = False # runs the RAG agent
SEED_CODE_GENERATION_AGENT = True # runs the seed data generation agent


# defining our rag tool input class
class RAGInput(BaseModel):
    query: str = Field(description="should be a search or retrieval query")

# defining out our rag tool function
@tool("rag-tool", args_schema=RAGInput, return_direct=True)
def chroma_rag(query: str) -> dict:
    """Extract similar texts the database"""
    # load up the db
    embedding_function = ChromaCodet5pEmbedding(CHECKPOINT)
    persistent_client = chromadb.PersistentClient() # default settings
    collection = persistent_client.get_collection("seed_code", embedding_function=embedding_function)

    # run the query
    results = collection.query(
    query_texts=[query], # Chroma will embed this for you
    n_results=2 # how many results to return
        )
    return results


if __name__ == "__main__":

    if RAG_AGENT:

        model = ChatOpenAI(model="gpt-4o")

        tools = [chroma_rag]

        # this creates the agent, should be in langgraph
        agent_executor = create_react_agent(model, tools)

        # invoking a call to the agent
        #response = agent_executor.invoke({"messages": [HumanMessage(content="Can you find similar text examples in the database to the #following text: {query}")]})
        #print(response["messages"])

        prompt_template = ChatPromptTemplate.from_messages([("user", "Can you find similar text examples in the database to the following text: {query}")])

        query = "def simulate_seir_model(N, I0, E0, R0, beta, gamma, sigma, days):"

        prompt = prompt_template.invoke({"query": query})

        # running it as a stream:
        for chunk in agent_executor.stream(prompt):
            print(chunk)
            print("---------")

    if SEED_CODE_GENERATION_AGENT:

        model = ChatOpenAI(model="gpt-4o")
        # testing it
        response = model.invoke([HumanMessage(content="Can you write a simple Python script to simulate covid spreading?")])
        print(response.content)

        # we now need to construct a loop to iterate over a number of key words in the prompt, lower the temperature,
        # and also perform output parsing of the results to collect write the scripts to disk properly. 