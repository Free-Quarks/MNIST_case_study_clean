# langchain version 0.2, 07/03/2024
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
import chromadb
from chromadb import Documents, EmbeddingFunction

from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from transformers import AutoModel, AutoTokenizer
from typing import List
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


from tree_encoder_model import GNNModel
from code_preprocessing import preprocess_tree_query
import torch

from vector_db import ChromaCodet5pEmbedding, ChromaTreeEmbedding

from tree_model_trainer import EMBED_DIM, IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, NODE_CLASSES, COMPRESSED_CLASSES, COMPRESSED_GRAPH_FEATURE, GRAPH_FEATURE


"""
Here we build out a library of functions/classes to perform ablation studies. 
There are two big classes in this case:
1) Dataset Generation Agent
    a) Instantiates a GPT-4o agent with specific settings
    b) Options: Seed vectordb, prompting framework, local diversity metric, output directory
2) Dataset Metrics
    b) Takes in a dataset directory and outputs a collection of metrics
"""

# this is class for our output parser
class PromptOutput(BaseModel):
    """how to format the rewritten prompt"""
    new_prompt: str = Field(description="should be the prompt that has been rewritten")

# defining our data generation output class
class DataOutput(BaseModel):
    """how to format the generated code"""
    code: str = Field(description="should be the code generated")

class Generator:
    def __init__(self, vectordb, prompting="few", metric="class", output="./dataset/default"):
        """
        vectordb: rag db for comparing similar data entries, this is a specific collection in question
        prompting: whether we are using zero or few shot prompting
        metric: the local metric used to compare data
        output: the output directory 
        termination_count: the amount of data to generate
        """
        self.vectordb = vectordb
        self.prompting = prompting
        self.metric = metric
        self.output = output
        self.termination_count = termination_count
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.7)

    def generate_prompt(self):
        """
        Generates a random prompt from our prompt taxonomy.

        returns the prompt and the class as a tuple: (prompt, class)
        """
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

        return (prompt, model_t)
    
    def generate_prompt_complex(self):
        """
        This using an LLM to rewrite the prompt in order to make it more stochastic and complicated.

        returns the more complex prompt
        """
        prompt, model_class = self.generate_prompt()
        model = self.model
        # defining the output parser
        parser = JsonOutputParser(pydantic_object=PromptOutput)
        format_instructions = parser.get_format_instructions()

        # define prompt template, random sampling of two different rewriting directions
        p = rd.randint(0,1)
        if p == 0:
            prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. {format_instructions}"),("human", "Rewrite the following prompt to produce more a complex model:\n{prompt}")])
        else:
            prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. {format_instructions}"),("human", "Rewrite the following prompt to be more open ended on the model it generates:\n{prompt}")])

        prompt_chain = prompt_template | model | parser

        result = prompt_chain.invoke({"prompt": prompt, "format_instructions": format_instructions})
        complex_prompt = result["new_prompt"]

        return (complex_prompt, model_class)

    def generate_data(self):
        """
        Generates a data point using a random prompt from our prompt taxonomy.

        returns the datapoint in question, code string
        """
        model = self.model
        prompt, model_class = self.generate_prompt_complex()

        data_parser = JsonOutputParser(pydantic_object=DataOutput)
        data_format_instructions = data_parser.get_format_instructions()

        data_prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. {data_format_instructions}"),("human", "{prompt}")])

        data_chain = data_prompt_template | model | data_parser
        data_result = data_chain.invoke({"data_prompt": prompt, "data_format_instructions": data_format_instructions})
        code_query = data_result['code']

        return (code_query, model_class)
    
    def compare_data_diversity(self, data_point, data_class):
        """
        Pull the similarity from the vectordb and runs our local diversity metrics.

        returns: the local diversity metric for the given data point and the most similar entries from our vectordb
        """
        if self.metric != "class":
            collection = self.vectordb

            results = collection.query(
                query_texts=[data_point], # Chroma will embed this for you
                n_results=3 # how many results to return
            )

            distances = results['distances'][0]
            avg_dist = np.mean(distances)

            # making an assumption here about the format of results
            data_entries = []
            for filename in results['metadata'][0]:
                with open(filename, 'r') as file:
                    content = file.read()
                file.close()
                data_entries.append(content)

            return (avg_dist, data_entries)

        else:
            return "temp"
    
    def few_shot_prompt(self, prompt, data, db_entries):
        """
        This takes in a previous prompt the data it generated and the similar db_entries and uses it to construct a 
        few-shot prompt to attempt to generate a more diverse output. 

        returns: another data point
        """

        return "temp"
    
    def data_writer(self, data, model_class):
        """
        Need to update the naming convention
        """
        num_genereated = len(os.listdir(self.output))
        with open(f"{self.output}/agentic-token-code-{model_class}-{num_genereated}.py", 'w') as f:
            print(data, file=f)
        f.close()