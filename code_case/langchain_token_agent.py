# langchain version 0.2, 07/03/2024
# for the agent
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.output_parsers import JsonOutputParser

# for the tool construction
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from typing import List
import numpy as np

# for rag, specifically chroma
import chromadb
from chromadb import Documents, EmbeddingFunction
from vector_db import ChromaCodet5pEmbedding, CHECKPOINT

# generic
import os
from time import sleep
import random as rd


# config
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DEVICE = "cuda"
WRITE_DIRECTORY = "./dataset/agentic_data_token"
NEW_DATA_TARGET = 100
DIVERSITY_THRESHOLD = 0.1

# defining the output parser structures
# RAG retrieval
class RAGOutput(BaseModel):
    """how to format responses from the database for the user"""
    distances: List[float] = Field(description="a list of distances or similarity measures from the retrieved entries")

# defining our rag tool input class
class RAGInput(BaseModel):
    """input query format to rag tool"""
    query: str = Field(description="should be a search or retrieval query")

# defining our diversity tool input class
class DiversityInput(BaseModel):
    """input query format to diversity tool"""
    distances: List[float] = Field(description="should be a list of distances")

# defining our data writer tool input class
class DataInput(BaseModel):
    """input format to write data to disk"""
    data_input: str = Field(description="should be the data to be written to disk")

# defining our prompt analysis output class
class PromptOutput(BaseModel):
    """how to format the rewritten prompt"""
    new_prompt: str = Field(description="should be the prompt that has been rewritten")

@tool("data-writer-tool", args_schema=DataInput)
def data_writer(data_input: str) -> None:
    """Writes data to disk."""
    num_genereated = len(os.listdir(WRITE_DIRECTORY))
    with open(f"agentic-token-code-{num_genereated}.py", 'w') as f:
        print(data_input, file=f)
    f.close()

@tool("termination-tool", args_schema=None, return_direct=True)
def termination() -> bool:
    """Determine if we should terminate data generation"""
    num_genereated = len(os.listdir(WRITE_DIRECTORY))
    if len(num_genereated) >= NEW_DATA_TARGET:
        return True
    else:
        return False

@tool("rag-tool", args_schema=RAGInput, return_direct=True)
def chroma_rag_token(query: str) -> List[float]:
    """Extract similar texts from the database"""
    # load up the db
    embedding_function = ChromaCodet5pEmbedding(CHECKPOINT)
    persistent_client = chromadb.PersistentClient() # default settings
    collection = persistent_client.get_collection("seed_code", embedding_function=embedding_function)

    # run the query
    results = collection.query(
    query_texts=[query], # Chroma will embed this for you
    n_results=2 # how many results to return
        )
    return results['distances'] # only need the distances

@tool("diversity-tool", args_schema=DiversityInput, return_direct=True)
def diversity_token(distances: List[float]) -> bool:
    """Determines weather the data generated is diverse enough to saved"""
    avg_dist = np.mean(distances)
    if avg_dist < DIVERSITY_THRESHOLD:
        return True
    else:
        return False

@tool("data-prompt-generator-tool", args_schema=None, return_direct=True)
def prompt_generator() -> str:
    """Creates a prompt to generate new data"""
    # original prompt taxonomy
    model_type = ["compartmental", "agent based", "network"]
    code_qualifier = ["incorrectly", "compactly", "verbosely", "normally"]
    model_list = ["SIR", "SEIR", "SEIRD", "SIDARTHE", "SEIRHD"]
    compart_method_list = ["Euler", "odeint", "RK2", "RK3", "RK4"]
    model_property = ["some form of stratification", "vaccination", "stratification by age", "stratification by sex", "no stratification"]
    
    # first we sample of prompt of the original prompt possibilities responsible for the seed data generation
    model = model_type[rd.randint(0, len(model_type)-1)]
    qualifier = code_qualifier[rd.randint(0, len(code_qualifier)-1)]
    prop = model_property[rd.randint(0, len(model_property)-1)]

    if model == "compartmental":
        method = compart_method_list[rd.randint(0, len(compart_method_list)-1)]
        model_class = model_list[rd.randint(0, len(model_list)-1)]
        prompt = f"Write a {model} model, {qualifier}, to simulate covid. It should be based on {model_class} and use the {method} method and include {prop}."
    else:
        prompt = f"Write a {model} model, {qualifier}, to simulate covid. It should include {prop}"

    # now we call an LLM to modify / diversify the prompt in a stocastic fashion
    sub_model = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    # defining the output parser
    sub_parser = JsonOutputParser(pydantic_object=PromptOutput)
    sub_format_instructions = parser.get_format_instructions()

    # define prompt template
    prompt_template = ChatPromptTemplate.from_messages([("system", "You are helpful assistant. {sub_format_instructions}"),("human", "Rewrite the following prompt to produce more a complex model:\n{prompt}")])

    # define the chain
    sub_chain = prompt_template | sub_model | sub_parser

    result = sub_chain.invoke({"prompt": prompt, "sub_format_instructions": sub_format_instructions})

    new_prompt = result["new_prompt"]

    return new_prompt

"""
This agent needs to be able to do the following:
O    1) Generate new data / code (prompt or subagentic tool or top level agent action)
X    2) RAG using the chroma vector db and generated data as a query into it (tool) [token specific]
X    3) Analysis the retrieved data metric information to determine if new data should 
        be generated (tool or top level control flow) [token specific]
X    4) Write a prompt to create more diverse data (tool or top level control flow)
-    5) Analysis of new prompt for accuracy of intended use and diversity (subagentic tool or top level control flow)
X    6) Write diverse data to disk (tool)
X    7) Determine when to terminate data generatation (tool or top level control flow)
"""

if __name__ == "__main__":

    model = ChatOpenAI(model="gpt-4o")
    tools = [chroma_rag_token] # adding a pydantic class as a tool is specific to openai

    # this creates the agent, should be in langgraph
    agent_executor = create_react_agent(model, tools)

    parser = JsonOutputParser(pydantic_object=RAGOutput)
    format_instructions = parser.get_format_instructions()

    prompt_template = ChatPromptTemplate.from_messages([("system", "You are helpful assistant. {format_instructions}"),("human", "Can you find similar examples in the database to the following\n{query}")])

    query = "def simulate_seir_model(N, I0, E0, R0, beta, gamma, sigma, days):"

    chain = prompt_template | agent_executor

    response = chain.invoke({"query": query, "format_instructions": format_instructions})

    output = parser.invoke(response["messages"][-1]) # really dumb you have to do this to get an agentic output parsed...

    print(output)