# langchain version 0.2, 07/03/2024
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from chromadb import Documents, EmbeddingFunction

from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from typing import List
import random as rd
import numpy as np
from enum import Enum
import pickle

import torch

from generator import ModelChecker, model_type_checker

if __name__ == "__main__":
    # set up the model and prompt to check the data
    test_label = False
    model = ChatOpenAI(model="gpt-4o", temperature=0.1) # lower temperature to be more deterministic for this task
    data_parser = JsonOutputParser(pydantic_object=ModelChecker)
    data_format_instructions = data_parser.get_format_instructions()
    data_prompt_template = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant. {data_format_instructions}"),("human", "Determine whether the give code is an implementation of a compartmental model or an agent-based model. \n Code:\n  {code}")])
    data_chain = data_prompt_template | model | data_parser

    data_directory = "./dataset/new_generator"
    file_checked_location = "./dataset/files_checked.pkl"

    skip_list = ["token_nomic_class_few", "token_nomic_class_few2", "token_nomic_class_zero", "token_nomic_class_zero2", "token_nomic_classless_few", "token_nomic_classless_few2", "token_llm_zero"]
    if os.path.exists(file_checked_location):
        with open(file_checked_location, 'rb') as file:
            file_checked_list = pickle.load(file)
            file.close()
    else:
        file_checked_list = ["place_holder.py"]
    for dirpath, _dirnames, filenames in os.walk(data_directory):
        old_subdir = ""
        for filename in filenames:
            subdir = dirpath.split("/")[-1]
            if subdir in skip_list:
                print(f"skipping subdirectory: {subdir}...")
                break
            elif filename.split(".")[-1] != "py":
                print(f"Skipping non-python ext: {filename}...")
                break
            else:
                if old_subdir != subdir:
                    print(f"Entering new subdirectory: {subdir}")
                unique_filename = subdir + "/" + filename
                if unique_filename not in file_checked_list:
                    p_label = filename.split("-")[-2]
                    index = filename.split("-")[-1]
                    m_type = filename.split("-")[1]
                    filelocation = dirpath + "/" + filename
                    with open(filelocation, 'r', encoding='utf-8') as file:
                        content = file.read()
                        file.close()
                    
                    data_result = data_chain.invoke({"code": content, "data_format_instructions": data_format_instructions})
                    output_label = data_result['model_type']
                    test_label, label = model_type_checker(output_label, p_label)
                    if not test_label:
                        print(f"Found miss classified data point:\p_label: {p_label} -> output_label: {output_label}")
                        new_filename = f"agentic-{m_type}-code-{label}-{index}"
                        final_path = dirpath + "/" + new_filename
                        print(final_path)
                        # write the new file
                        with open(final_path, 'w') as f:
                            print(content, file=f)
                            f.close()
                        # now to delete the old data
                        os.remove(filelocation)
                        unique_filename = subdir + "/" + new_filename
                        file_checked_list.append(unique_filename)
                    else:
                        file_checked_list.append(unique_filename)

                    print("Recording newly checked files...")
                    with open(file_checked_location, 'wb') as file:
                        pickle.dump(file_checked_list, file)
                        file.close()

            old_subdir = subdir


                