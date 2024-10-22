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
from enum import Enum


from tree_encoder_model import GNNModel
from code_preprocessing import preprocess_tree_query
import torch

from vector_db import ChromaCodet5pEmbedding, ChromaTreeEmbedding, ChromaNomicEmbedding

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

# defining our data generation output class
class LLMOutput(BaseModel):
    """How to format about if a give code script is sufficiently different from other given examples"""
    answer: bool = Field(description="boolean of True if it is diverse enough and False if not diverse enough")
    suggestion: str = Field(description="suggestion of how to change the code to make it more diverse")

class ModelType(str, Enum):
    """Enum of the avialable model types, being only compartmental or agent-based."""
    compartmental = "compartmental"
    abm = "agent-based"

class ModelChecker(BaseModel):
    """How to format which type of epidemiology model the give code example is an implemenation of"""
    model_type: ModelType = Field(description="This is the model type of the code example. Should be either compartmental or agent-based")

class Generator:
    def __init__(self, vectordb, prompting="few", metric="class", threshold=0.3, output="./dataset/default"):
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
        self.threshold = threshold # this is default of 0.3 for token class metrics, 0.35 for token classless metrics
        self.output = output
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
            model_t = "compart"
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
        try:
            data_result = data_chain.invoke({"prompt": prompt, "data_format_instructions": data_format_instructions})
        except:
            data_result = data_chain.invoke({"prompt": prompt, "data_format_instructions": data_format_instructions})
        code_query = data_result['code']

        return (code_query, model_class, prompt)
    
    def compare_data_diversity(self, data_point, data_class):
        """
        Pull the similarity from the vectordb and runs our local diversity metrics.

        returns: the local diversity metric for the given data point and the most similar entries from our vectordb
        """
        collection = self.vectordb

        if self.metric == "classless":

            results = collection.query(
                query_texts=[data_point], # Chroma will embed this for you
                n_results=3 # how many results to return
            )

            distances = results['distances'][0]
            avg_dist = np.mean(distances)

            # making an assumption here about the format of results
            data_entries = []
            for source in results['metadatas'][0]:
                filename = source['source']
                with open(filename, 'r') as file:
                    content = file.read()
                file.close()
                data_entries.append(content)

            llm_rewrite = ""

            return (avg_dist, data_entries, llm_rewrite)
        
        elif self.metric == "llm":
            results = collection.query(
                query_texts=[data_point], # Chroma will embed this for you
                n_results=2 # how many results to return
            )

            #distances = results['distances'][0]
            #avg_dist = np.mean(distances)

            # making an assumption here about the format of results
            data_entries = []
            for source in results['metadatas'][0]:
                filename = source['source']
                with open(filename, 'r') as file:
                    content = file.read()
                file.close()
                data_entries.append(content)
            
            # use the pulled similar code examples and let the LLM decide if they are different enough
            model = self.model

            data_parser = JsonOutputParser(pydantic_object=LLMOutput)
            data_format_instructions = data_parser.get_format_instructions()
            code_parser = JsonOutputParser(pydantic_object=DataOutput)
            code_format_instructions = code_parser.get_format_instructions()

            prompt = f"If the goal is to test a code-based NLP model, is the following code example sufficiently different from the following other examples we are currently using to test the model and what suggestions would you give to modify the example to make it more diverse from the currently used examples? \n\n New Example:\n{data_point}\n\n Previous/Current Examples: \n{data_entries[0]}\n\n {data_entries[1]}"

            data_prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. {data_format_instructions}"),("human", "{prompt}")])

            data_chain = data_prompt_template | model | data_parser
            data_result = data_chain.invoke({"prompt": prompt, "data_format_instructions": data_format_instructions})
            answer = data_result['answer']
            suggestions = data_result['suggestion']
            print("*******")
            print(f"New code is diverse?: {answer}")
            print(f"{suggestions}")

            if answer:
                avg_dist = 1
                llm_rewrite = ""
            else:
                avg_dist = 0.75 # this let's us know if we rewrote the code
                prompt = f"Rewrite the following code with the given suggestions, but ignore any suggestions that would change the base type of model. \n\n Code:\n{data_point}\n\n Suggestions: \n{suggestions}"
                code_chain = data_prompt_template | model | code_parser
                code_result = code_chain.invoke({"prompt": prompt, "data_format_instructions": code_format_instructions})
                llm_rewrite = code_result['code']

            return (avg_dist, data_entries, llm_rewrite)

        else:
            n_results = 250
            num_class = 0
            while num_class < 3:
                num_class = 0
                results = collection.query(
                    query_texts=[data_point], # Chroma will embed this for you
                    n_results=n_results # how many results to return
                )

                relevant_data = []
                for (i, source) in enumerate(results['metadatas'][0]):
                    filename = source['source']
                    if filename.split("-")[-2] == data_class:
                        relevant_data.append(i)
                
                num_class = len(relevant_data)
                n_results += 25
                if n_results > 500:
                    num_class = 5
                    relevant_data = [1, 2, 3, 4]
                    print(f"Error: {data_class}")

            distances = []
            data_entries = []
            relevant_data = relevant_data[:3]
            for index in relevant_data:        
                distances.append(results['distances'][0][index])
                source = results['metadatas'][0][index]
                filename = source['source']
                with open(filename, 'r') as file:
                    content = file.read()
                file.close()
                data_entries.append(content)

            avg_dist = np.mean(distances)

            print(f"logging: {data_class}")

            llm_rewrite = ""

            return (avg_dist, data_entries, llm_rewrite)
    
    def few_shot_generate_data(self, prompt, data, db_entries):
        """
        This takes in a previous prompt the data it generated and the similar db_entries and uses it to construct a 
        few-shot prompt to attempt to generate a more diverse output. 

        returns: another data point
        """
        model = self.model

        example_0 = f"{db_entries[0]}"
        example_1 = f"{db_entries[1]}"
        example_2 = f"{db_entries[2]}"

        data_parser = JsonOutputParser(pydantic_object=DataOutput)
        format_instructions = data_parser.get_format_instructions()

        # since we are using the few-shot approach not how it's intended to be used, we will just use a 
        # more a generic prompt template and modify it
        few_shot_prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. {format_instructions}"),("human", "Please make sure the code you generate is sufficently different from these four examples: {example_0}\n\n{example_1}\n\n{example_2}\n\n{data}\n\n{prompt}")])

        few_shot_prompt_chain = few_shot_prompt_template | model | data_parser

        # invoke the chain
        try:
            data_result = few_shot_prompt_chain.invoke({"example_0": example_0, "example_1": example_1, "example_2": example_2, "data": data, "prompt": prompt, "format_instructions": format_instructions})
        except:
            data_result = few_shot_prompt_chain.invoke({"example_0": example_0, "example_1": example_1, "example_2": example_2, "data": data, "prompt": prompt, "format_instructions": format_instructions})            
        code_query = data_result['code']

        return code_query
    
    def data_writer(self, data, model_class):
        """
        Need to update the naming convention
        """
        num_genereated = len(os.listdir(self.output))
        with open(f"{self.output}/agentic-tree-code-{model_class}-{num_genereated}.py", 'w') as f:
            print(data, file=f)
        f.close()

    def generation_pass(self):
        """
        This performs one pass of data generation based on given arguments. This is includes a final test to make sure the labels are correct as a final sanity check. The compounding stocastic responses from the LLM can result in label drift on occasion. 

        Inputs: (self)
        Returns: 
            avg_dist (float): the distance the data point is from it's nearest neighbor in the seed dataset
            few_avg_dist (float): the distance the data point is from it's nearest neighbors, if few-shot is run, else is 0
        """
        # this is for label sanity checks
        test_label = False
        model = self.model
        data_parser = JsonOutputParser(pydantic_object=ModelChecker)
        data_format_instructions = data_parser.get_format_instructions()
        data_prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. {data_format_instructions}"),("human", "Determine whether the give code is an implementation of a compartmental model or an agent-based model. \n Code:\n  {code}")])
        data_chain = data_prompt_template | model | data_parser

        avg_dist = 0
        few_avg_dist = 0
        data_point, data_class, prompt = self.generate_data()
        avg_dist, data_entries, llm_rewrite = self.compare_data_diversity(data_point, data_class)
        print(f"zero-shot distance: {avg_dist}")
        if llm_rewrite == "":
            data_result = data_chain.invoke({"code": data_point, "data_format_instructions": data_format_instructions})
            model_type = data_result['model_type']
            test_label, label = model_type_checker(model_type, data_class)
        else:
            data_result = data_chain.invoke({"code": llm_rewrite, "data_format_instructions": data_format_instructions})
            model_type = data_result['model_type']
            test_label, label = model_type_checker(model_type, data_class)
        if avg_dist >= self.threshold and test_label:
            if llm_rewrite == "":
                self.data_writer(data_point, data_class)
            else:
                self.data_writer(llm_rewrite, data_class)
        elif avg_dist >= self.threshold and not test_label:
            if llm_rewrite == "":
                self.data_writer(data_point, label)
            else:
                self.data_writer(llm_rewrite, label)
        if avg_dist < self.threshold and self.prompting == "few":
            few_data_point = self.few_shot_generate_data(prompt, data_point, data_entries)
            few_avg_dist, _few_data_entries, _llm_rewrite = self.compare_data_diversity(few_data_point, data_class)
            print(f"few-shot distance: {few_avg_dist}")
            data_result = data_chain.invoke({"code": few_data_point, "data_format_instructions": data_format_instructions})
            model_type = data_result['model_type']
            test_label, label = model_type_checker(model_type, data_class)
            if few_avg_dist >= self.threshold and test_label:
                self.data_writer(few_data_point, data_class)
            elif few_avg_dist >= self.threshold and not test_label:
                self.data_writer(few_data_point, label)
        if not test_label:
            print(f"Had model type swap at some point:\ndata_class: {data_class} -> model_type: {model_type}")
        return avg_dist, few_avg_dist

# checks a returned enum of the LLM's response of the model type to the ground truth string of the model class
def model_type_checker(output, truth):
    test_label = False
    if output == ModelType.compartmental and truth == "compart":
        test_label = True
        label = truth
    elif output == ModelType.abm and truth == "abm":
        test_label = True
        label = truth
    else:
        if output == ModelType.abm:
            label = "abm"
        if output == ModelType.compartmental:
            label = "compart"
    
    return test_label, label


if __name__ == "__main__":
    write_directory = "./dataset/new_generator/token_nomic_classless_few2"
    few_vs_zero_logs = write_directory + "/distances.npz"
    checkpoint = "Salesforce/codet5p-110m-embedding"
    nomic_checkpoint = "nomic-ai/nomic-embed-text-v1.5"
    tree_checkpoint = "./models/tree_ae/tree_ae.pth"
    url = "http://localhost:8000/code2fn/fn-given-filepaths"
    matryoshka_dim = 128

    ####embedding_function_chroma_codet = ChromaCodet5pEmbedding(checkpoint)
    ####embedding_function_chroma_tree = ChromaTreeEmbedding(checkpoint, tree_checkpoint, url)
    embedding_function_chroma_nomic = ChromaNomicEmbedding(nomic_checkpoint, matryoshka_dim)
    persistent_client_tok = chromadb.PersistentClient() # default settings
    # this gets the collection since it's already present
    vectordb_seed = persistent_client_tok.get_collection("seed_code_nomic_fixed", embedding_function=embedding_function_chroma_nomic)
    # need to check that the data is being pulled correctly from the metadata in the vectordb
    # need to check the few shot prompt is staying consistent with the class and overall prompt
    # metrics: class, classless, llm | prompting: zero, few
    generator = Generator(vectordb_seed, prompting="few", metric="classless", threshold=0.1, output=write_directory)

    zero_shot_distances = []
    few_shot_distances = []
    print("-----------------")
    print(f"{len(os.listdir(write_directory))} / 200 files generated")
    print("-----------------")
    run_counter = 0
    while len(os.listdir(write_directory)) < 200:
        avg_dist, few_avg_dist = generator.generation_pass()
        if avg_dist != 0:
            zero_shot_distances.append(avg_dist)
        if few_avg_dist != 0:
            few_shot_distances.append(few_avg_dist)
        run_counter += 1

        # this is to make sure we update the few verse zero logs periodically in case we get a crash
        if run_counter % 4 == 0:
            if os.path.exists(few_vs_zero_logs):
                data = np.load(few_vs_zero_logs)
                old_zero_array = data['array1']
                old_few_array = data['array2']
                new_zero_array = np.array(zero_shot_distances)
                new_few_array = np.array(few_shot_distances)
                zero_shot = np.concatenate((old_zero_array, new_zero_array))
                few_shot = np.concatenate((old_few_array, new_few_array))
                np.savez(few_vs_zero_logs, array1=zero_shot, array2=few_shot)
                # now to clear the vectors since we dumped the content, also so we don't double count
                zero_shot_distances = []
                few_shot_distances = []
            else:
                zero_shot = np.array(zero_shot_distances)
                few_shot = np.array(few_shot_distances)
                np.savez(few_vs_zero_logs, array1=zero_shot, array2=few_shot)

    # Load the .npz file
    data = np.load(few_vs_zero_logs)

    # Access the arrays using their assigned names
    array1 = data['array1']
    array2 = data['array2']
    print(np.mean(array1))
    print(len(array1))
    print(len(array2))

