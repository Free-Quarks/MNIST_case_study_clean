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

# for rag, specifically chroma
import chromadb
from chromadb import Documents, EmbeddingFunction
from vector_db import ChromaCodet5pEmbedding, CHECKPOINT

# generic
import os
from time import sleep

# local
from code_preprocessing import code_2_fn

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
SEED_CODE_GENERATION_AGENT = False # runs the seed data generation agent
SEED_FN_GRAPH_GENERATION = True # for converting the code into fn's and graphs

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

        parser = JsonOutputParser(pydantic_object=RAGOutput)
        format_instructions = parser.get_format_instructions()

        prompt_template = ChatPromptTemplate.from_messages([("system", "You are helpful assistant. {format_instructions}"),("human", "Can you find similar examples in the database to the following\n{query}")])

        query = "def simulate_seir_model(N, I0, E0, R0, beta, gamma, sigma, days):"

        chain = prompt_template | agent_executor

        response = chain.invoke({"query": query, "format_instructions": format_instructions})

        output = parser.invoke(response["messages"][-1]) # really dumb you have to do this to get an agentic output parsed...

        print(output)

    if SEED_CODE_GENERATION_AGENT:

        model = ChatOpenAI(model="gpt-4o", temperature=0.7)

        # number of full generation sequences to run
        num_runs = 4 # each run should make 200-248 code scripts depending on fail rate, 248 is no failures

        # common key words
        programmer_type = ["college student", "software engineer"]
        code_qualifier = ["Incorrectly", "compactly", "verbosely", ""]

        # model type key words
        model_type = ["compartmental", "agent based", "network"]
        
        # sub model key words 
        # compartimental
        model_list = ["SIR", "SEIR", "SEIRD", "SIDARTHE", "SEIRHD"]
        method_list = ["Euler", "odeint", "RK2", "RK3", "RK4"]

        # agentic and network
        model_property = ["some form of stratification", "vaccination", "stratification by age", "stratification by sex"]

        # defining the output parser
        parser = JsonOutputParser(pydantic_object=CodeOutput)
        format_instructions = parser.get_format_instructions()

        compart_prompt_template = ChatPromptTemplate.from_messages([("human", "Write a {model_type} model {code_qualifier}. It should be based on {model_list} and use the {method_list}."), ("system", "You are a {programmer_type} that writes python scripts to simulate covid based on how the user requests and {format_instructions}")])

        abm_prompt_template = ChatPromptTemplate.from_messages([("human", "Write a {model_type} model to simulate covid, {code_qualifier}. It should include {model_property}"), ("system", "You are a {programmer_type} that writes python scripts to simulate covid based on user requests. {format_instructions}")])

        chain = compart_prompt_template | model | parser    

        abm_chain = abm_prompt_template | model | parser

        compart_counter = 0
        abm_counter = 0

        code_directory = "./dataset/new_test_code"
        fn_directory = "./dataset/new_test_fn"
        for _i in range(num_runs):
            for programmer in programmer_type:
                for qualifier in code_qualifier:
                    for model in model_type:
                        if model == "compartmental": 
                            for submodel in model_list:
                                for method in method_list:
                                    compart_counter += 1
                                    try:
                                        result = chain.invoke({"model_type": model, "code_qualifier": qualifier, "model_list": submodel, "method_list": method, "programmer_type": programmer, "format_instructions": format_instructions})

                                        with open(f"{code_directory}/output-code-compart-{compart_counter}.py", 'w') as f:
                                            print(result['code'], file=f)
                                        f.close()
                                    except:
                                        print(f"chain invoke failed on compartmental model: {compart_counter}")
                                    sleep(1) # to help manage calls per minute limits 
                        else:
                            # this is currently not working, all of these abm prompts failed
                            for model_prop in model_property:
                                abm_counter += 1
                                try:
                                    result = abm_chain.invoke({"model_type": model, "code_qualifier": qualifier, "model_property": model_prop, "programmer_type": programmer, "format_instructions": format_instructions})

                                    with open(f"{code_directory}/output-code-abm-{abm_counter}.py", 'w') as f:
                                        print(result['code'], file=f)
                                    f.close()
                                except:
                                    print(f"chain invoke failed on abm model: {abm_counter}\nkeyworkds: {qualifier}, {model_prop}, {programmer}\n---------")
                                sleep(1) # to help manage calls per minute limits 
    if SEED_FN_GRAPH_GENERATION:
        # now for converting the new code into function networks as well
        
        # url for the code2fn service in skema (service needs to be up and running)
        url = "http://0.0.0.0:8000/code2fn/fn-given-filepaths" # check this

        code_directory = "./dataset/new_test_code"
        fn_directory = "./dataset/new_test_fn"

        code_2_fn(code_directory, fn_directory, url)
