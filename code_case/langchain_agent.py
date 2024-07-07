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

# defining the output parser structures
# RAG retrieval
class RAGOutput(BaseModel):
    """how to format responses from the database for the user"""
    distances: List[float] = Field(description="a list of distances or similarity measures from the retrieved entries")
    documents: List[str] = Field(description="a list of the documents from the retrieved entries")

# seed code generation
class CodeOutput(BaseModel):
    """Code to provide to the user"""
    code: str = Field(description="the epidimiology code generated")

# defining our rag tool input class
class RAGInput(BaseModel):
    """input query format to rag tool"""
    query: str = Field(description="should be a search or retrieval query")

# defining out our rag tool function
@tool("rag-tool", args_schema=RAGInput, return_direct=True)
def chroma_rag(query: str) -> dict:
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
    return results


if __name__ == "__main__":

    if RAG_AGENT:

        model = ChatOpenAI(model="gpt-4o")

        tools = [chroma_rag] # adding a pydantic class as a tool is specific to openai

        # this creates the agent, should be in langgraph
        agent_executor = create_react_agent(model, tools)

        #model.bind_tools(tools)

        parser = JsonOutputParser(pydantic_object=RAGOutput)
        format_instructions = parser.get_format_instructions()

        # defining the output parser, specific to openai
        #parser = JsonOutputToolsParser()

        prompt_template = ChatPromptTemplate.from_messages([("system", "You are helpful assistant. {format_instructions}"),("human", "Can you find similar examples in the database to the following\n{query}")])

        query = "def simulate_seir_model(N, I0, E0, R0, beta, gamma, sigma, days):"

        chain = prompt_template | agent_executor

        response = chain.invoke({"query": query, "format_instructions": format_instructions})

        output = parser.invoke(response["messages"][-1]) # really dumb you have to do this to get an agentic output parsed...

        print(output)

    if SEED_CODE_GENERATION_AGENT:

        model = ChatOpenAI(model="gpt-4o", temperature=0.7)
        # testing it
        # response = model.invoke([HumanMessage(content="Can you write a simple Python script to simulate covid spreading?")])
        # print(response.content)

        # we now need to construct a loop to iterate over a number of key words in the prompt, lower the temperature,
        # and also perform output parsing of the results to collect write the scripts to disk properly. 

        # common key words
        programmer_type = ["college student", "software engineer"]
        code_qualifier = ["Incorrectly", "compactly", "verbosely", ""]

        # model type key words
        model_type = ["compartimental", "agentic", "network"]
        
        # sub model key words 
        # compartimental
        model_list = ["SIR", "SEIR", "SEIRD", "SIDARTHE", "SEIRHD"]
        method_list = ["Euler", "odeint", "RK2", "RK3", "RK4"]


        # defining the output parser
        parser = JsonOutputParser(pydantic_object=CodeOutput)
        format_instructions = parser.get_format_instructions()

        # agentic and network
        model_properity = ["vaccination", "stratification by age", "stratification by sex"]

        compart_prompt_template = ChatPromptTemplate.from_messages([("human", "Write a {model_type} model {code_qualifier}. It should be based on {model_list} and use the {method_list}."), ("system", "You are a {programmer_type} that writes python scripts to simulate covid based on how the user requests and {format_instructions}")])

        abm_prompt_template = ChatPromptTemplate.from_messages([("human", "Write a {model_type} model {code_qualifier}. It should include {model_property}"), ("system", "You are a {programmer_type} that writes python scripts to simulate covid based on how the user requests. {format_instructions}")])

        # invoke the chain

        chain = compart_prompt_template | model | parser

        result = chain.invoke({"model_type": model_type[0], "code_qualifier": code_qualifier[0], "model_list": model_list[0], "method_list": method_list[0], "programmer_type": programmer_type[0], "format_instructions": format_instructions})

        print(result)