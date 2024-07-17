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
from vector_db import ChromaCodet5pEmbedding, CHECKPOINT

# generic
import os
from time import sleep
import random as rd
import pprint


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

# defining our data writer tool input class
class DataInput(BaseModel):
    """input format to write data to disk"""
    data_input: str = Field(description="should be the data to be written to disk")

# defining our data generation tool input class
class GenInput(BaseModel):
    """input format to write data to disk"""
    gen_input: str = Field(description="this can be any short string")

# defining our prompt analysis output class
class PromptOutput(BaseModel):
    """how to format the rewritten prompt"""
    new_prompt: str = Field(description="should be the prompt that has been rewritten")

# defining our data generation output class
class DataOutput(BaseModel):
    """how to format the generated code"""
    code: str = Field(description="should be the code generated")

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

@tool("diversity-tool", args_schema=RAGInput, return_direct=True)
def chroma_rag_token_diversity(query: str) -> bool:
    """Determines weather a given data point is diverse from the existing database."""
    # load up the db
    embedding_function = ChromaCodet5pEmbedding(CHECKPOINT)
    persistent_client = chromadb.PersistentClient() # default settings
    collection = persistent_client.get_collection("seed_code", embedding_function=embedding_function)

    # run the query
    results = collection.query(
    query_texts=[query], # Chroma will embed this for you
    n_results=2 # how many results to return
        )
    
    # compare the distances
    distances = results['distances'][0]
    avg_dist = np.mean(distances)
    if avg_dist < DIVERSITY_THRESHOLD:
        return True
    else:
        return False

@tool("data-generator-tool", args_schema=GenInput, return_direct=True)
def data_generator(gen_input: str) -> str:
    """Generates new data"""
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
    sub_format_instructions = sub_parser.get_format_instructions()

    # define prompt template
    prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. {sub_format_instructions}"),("human", "Rewrite the following prompt to produce more a complex model:\n{prompt}")])

    # define the chain
    sub_chain = prompt_template | sub_model | sub_parser

    result = sub_chain.invoke({"prompt": prompt, "sub_format_instructions": sub_format_instructions})

    # now to run the new prompt through to get the data/code
    data_prompt = result["new_prompt"]

    data_parser = JsonOutputParser(pydantic_object=DataOutput)
    data_format_instructions = data_parser.get_format_instructions()

    data_prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. {data_format_instructions}"),("human", "{data_prompt}")])

    data_chain = data_prompt_template | sub_model | data_parser

    data_result = data_chain.invoke({"data_prompt": data_prompt, "data_format_instructions": data_format_instructions})

    return data_result['code']

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

    # rag tools for this agent
    tools = [chroma_rag_token_diversity, data_writer, termination, data_generator] 

    prompt = hub.pull("hwchase17/react")

    # this creates the agent, should be in langgraph
    agent = create_react_agent(model, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    #agent_executor.invoke({"input": "Generate new data that is more diverse than what is in our database."})

    """"
    Agentic Flow should be as follows: 
        - Generate new data point (tool call action)
            - This calls a sub-agent, which will rewrite the prompt, then ingest the prompt itself to create the new data point
        - Checks if the data is diverse
        - if diverse it saves the data to disk
        - checks if it has generated enough data
    """

    for chunk in agent_executor.stream({"input": "Generate new data and then compare its diversity to our database. If it is more diverse, write the data to disk."}):
        print("------")
        pprint.pprint(chunk, depth=1)