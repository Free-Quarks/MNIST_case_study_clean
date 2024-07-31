# langchain version 0.2, 07/03/2024
# for the agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.output_parsers import JsonOutputParser
from langchain import hub

# for the tool construction
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from typing import List
import numpy as np

# for rag, specifically chroma
import chromadb
from chromadb import Documents, EmbeddingFunction
from vector_db import ChromaTreeEmbedding, CHECKPOINT, TREE_CHECKPOINT, URL

# generic
import os
from time import sleep
import random as rd


# config
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DEVICE = "cuda"
WRITE_DIRECTORY = "./dataset/agentic_data_tree"
NEW_DATA_TARGET = 100
DIVERSITY_THRESHOLD = 0.001

# defining the output parser structures
# defining our prompt analysis output class
class PromptOutput(BaseModel):
    """how to format the rewritten prompt"""
    new_prompt: str = Field(description="should be the prompt that has been rewritten")

# defining our data generation output class
class DataOutput(BaseModel):
    """how to format the generated code"""
    code: str = Field(description="should be the code generated")

def data_writer(data_input: str, model_t: str) -> None:
    """Writes data to disk."""
    num_genereated = len(os.listdir(WRITE_DIRECTORY))
    with open(f"{WRITE_DIRECTORY}/agentic-tree-code-{model_t}-{num_genereated}.py", 'w') as f:
        print(data_input, file=f)
    f.close()

def termination() -> bool:
    """Determine if we should terminate data generation"""
    num_genereated = len(os.listdir(WRITE_DIRECTORY))
    if num_genereated >= NEW_DATA_TARGET:
        return True
    else:
        return False

def chroma_rag_tree_diversity(query: str) -> tuple[bool, float]:
    """Determines weather a given data point is diverse from the existing database."""
    # load up the db
    embedding_function = ChromaTreeEmbedding(CHECKPOINT, TREE_CHECKPOINT, URL)
    persistent_client = chromadb.PersistentClient() # default settings
    collection = persistent_client.get_collection("seed_code_tree", embedding_function=embedding_function)

    # run the query
    results = collection.query(
    query_texts=[query], # Chroma will embed this for you
    n_results=3 # how many results to return
        )
    # compare the distances
    distances = results['distances'][0]
    avg_dist = np.mean(distances)
    if avg_dist > DIVERSITY_THRESHOLD:
        return True, avg_dist
    else:
        return False, avg_dist


if __name__ == "__main__":

    while not termination():
        # original prompt taxonomy
        model_type = ["compartmental", "agent-based", "network"]
        code_qualifier = ["incorrectly", "compactly", "verbosely", "normally"]
        model_list = ["SIR", "SEIR", "SEIRD", "SIDARTHE", "SEIRHD"]
        compart_method_list = ["Euler", "odeint", "RK2", "RK3", "RK4"]
        model_property = ["some form of stratification", "vaccination", "stratification by age", "stratification by sex", "no stratification"]
        
        # first we sample of prompt of the original prompt possibilities responsible for the seed data generation
        model_t = model_type[rd.randint(0, len(model_type)-1)]
        qualifier = code_qualifier[rd.randint(0, len(code_qualifier)-1)]
        prop = model_property[rd.randint(0, len(model_property)-1)]

        if model_t == "compartmental":
            method = compart_method_list[rd.randint(0, len(compart_method_list)-1)]
            model_class = model_list[rd.randint(0, len(model_list)-1)]
            prompt = f"Write a {model_t} model, {qualifier}, to simulate covid. It should be based on {model_class} and use the {method} method and include {prop}."
        else:
            prompt = f"Write a {model_t} model, {qualifier}, to simulate covid. It should include {prop}"
            model_t = "abm"

        # now we call an LLM to modify / diversify the prompt in a stocastic fashion
        model = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        # defining the output parser
        parser = JsonOutputParser(pydantic_object=PromptOutput)
        format_instructions = parser.get_format_instructions()

        # define prompt template, random sampling of two different rewriting directions
        p = rd.randint(0,1)
        if p == 0:
            prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. {format_instructions}"),("human", "Rewrite the following prompt to produce more a complex model:\n{prompt}")])
        else:
            prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. {format_instructions}"),("human", "Rewrite the following prompt to be more open ended on the model it generates:\n{prompt}")])
        
        # define the chain
        sub_chain = prompt_template | model | parser

        result = sub_chain.invoke({"prompt": prompt, "format_instructions": format_instructions})

        # now to run the new prompt through to get the data/code
        data_prompt = result["new_prompt"]

        data_parser = JsonOutputParser(pydantic_object=DataOutput)
        data_format_instructions = data_parser.get_format_instructions()

        data_prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. {data_format_instructions}"),("human", "{data_prompt}")])

        data_chain = data_prompt_template | model | data_parser

        data_result = data_chain.invoke({"data_prompt": data_prompt, "data_format_instructions": data_format_instructions})

        code_query = data_result['code']

        # now we test if it's diverse
        diverse, quantification = chroma_rag_tree_diversity(code_query)

        if diverse:
            data_writer(code_query, model_t)
            print("---------")
            print(f"one pass, the data was this diverse: {quantification} and was written to disk, p: {p}")
            print("---------")
        else:
            print("---------")
            print(f"one pass, the data was this diverse: {quantification} and was not written to disk, p: {p}")
            print("---------")
        
        
