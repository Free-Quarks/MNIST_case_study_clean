from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import requests
import json

# run the following if it complains: >$ export TOKENIZERS_PARALLELISM="true"

# Dataset class for to wrap our data
class ListDatasetWrapper(Dataset):
    def __init__(self, data_list):
        """Initialize the dataset with a Python list"""
        self.data_list = data_list

    def __len__(self):
        """Return the total number of data points in the dataset"""
        return len(self.data_list)

    def __getitem__(self, idx):
        """Retrieve an entry by index that includes only the data, no label"""
        data = self.data_list[idx]
        return data


def preprocess_tokenized(directory):
    """
    This function takes in the directory of the code that will be processed into a dataset of tokenized sequence inputs.
    These will also be broken into the length of the contextual window required for our embedding model (600 tokens). 

    Args:
        directory (string): the string of the directory of the code to be processed
    Returns:
        dataset (Dataset): a dataset of processed code
    """
    # initialize encoder for tokenization
    checkpoint = "Salesforce/codet5p-110m-embedding"
    device = "cuda:0"  # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    # first we read in and encode all the files in the directory
    file_encodings = []
    walker = os.walk(directory)
    for root, _dirs, files in walker:
        for file in files:
            file_path = root+"/"+file
            file_content = ""
            with open(file_path, 'r') as file:
                file_content = file.read()
                file.close()

            # not with the truncation=True argument set as is, this will truncate larger code files to the max possible size, 512 tokens. So any 512 token inputs are likely truncated. 
            file_encoding = tokenizer.encode(file_content, truncation=True, return_tensors="pt").to(device)
            print(file_encoding.size())
            file_encodings.append(file_encoding)
            
    # wrap our list into a dataset
    dataset = ListDatasetWrapper(file_encodings)
    return dataset


def code_2_fn(code_directory, fn_directory, url):
    """
    This function converts a collection of source code in the code_directory into function networks through an
    API call to the url and then writes them out to the fn_directory. 

    Args:
        code_directory (string): The directory of the source code to be walked over. [note: no ending backslash]
        fn_directory (string): The directory that the function networks will be written out to. [note: no ending backslash]
        url (string): the url for our code2fn service
    Returns:
        none: This function writes out to disk and doesn't have a return, unless it errors out
    """
    for filename in os.listdir(code_directory):
        f = os.path.join(directory, filename)

        try:
            single_snippet_payload = {
                    "files": [filename],
                    "blobs": [open(f).read()],
                    "dependency_depth": 0
                }
            
            response = requests.post(url, json=single_snippet_payload)
            print(response)
            #print(response.json())
            gromet = json.dumps(response.json(), indent=4)

            gromet_json = json.loads(gromet)
            try:
                with open(fn_directory+f"/{filename[:-3]}.json", "w") as outfile:
                    gromet_collection_dict = (
                                    gromet_json
                                )
                    outfile.write(
                        dictionary_to_gromet_json(
                            del_nulls(gromet_collection_dict), level=0
                            )
                        )
            except:
                gromet_collection_dict = (gromet_json)
                with open(fn_directory+f"/{filename[:-3]}.json", "w") as outfile:
                    outfile.write(f"{dictionary_to_gromet_json(del_nulls(gromet_collection_dict), level=0)}")
        except:
            print(f"{f} failed to process\n")

def preprocess_tree(directory):
    """
    This function takes in the directory of the code to be processed into a dataset of code trees. Each node will have a subet of tokens and the feature vector for that node will come from the embedding model we are using. 

    Args:
        directory (string): the string of the directory of code to be processed
    Returns:
        dataset (Dataset): a dataset of processed code
    """
    # There are 3 different aspects of information we want to encode in each node:
    #       1) The embedding vector of the tokens in the node
    #       2) The positional encodings using eigenvectors of the Laplacian of the graph or other options
    #       3) The node type embedding as well, function, literal, etc. 

    # 1) and 2) will be added together as is usual for the embedding and positional information in transformer architectures
    # 3) is still being thought about, but perhaps will be a seperate dimension of the feature space. 


    dataset = []
    return dataset

def dictionary_to_gromet_json(
    o, fold_level=5, indent=4, level=0, parent_key=""
):
    """
    This is just to make the json more human readable if wanted
    """
    if level < fold_level:
        newline = "\n"
        space = " "
    else:
        newline = ""
        space = ""
    ret = ""
    if isinstance(o, str):
        ret += json.dumps(
            o
        )  # json.dumps() will properly escape special characters
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, float):
        ret += "%.7g" % o
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, list):
        ret += "[" + newline
        comma = ""
        for e in o:
            ret += comma
            comma = "," + newline
            ret += space * indent * (level + 1)
            ret += dictionary_to_gromet_json(
                e, fold_level, indent, level + 1, parent_key
            )
        ret += newline + space * indent * level + "]"
    elif isinstance(o, dict):
        ret += "{" + newline
        comma = ""
        for k, v in o.items():
            ret += comma
            comma = "," + newline
            ret += space * indent * (level + 1)
            ret += '"' + str(k) + '":' + space
            if k == "fn":
                ret += dictionary_to_gromet_json(v, 4, indent, level + 1, k)
            elif k == "attributes":
                ret += dictionary_to_gromet_json(v, 6, indent, level + 1, k)
            elif k == "bf" and parent_key == "fn":
                ret += dictionary_to_gromet_json(v, 5, indent, level + 1, k)
            elif k == "bf" and parent_key == "value":
                ret += dictionary_to_gromet_json(v, 7, indent, level + 1, k)
            else:
                ret += dictionary_to_gromet_json(
                    v, fold_level, indent, level + 1, k
                )
        ret += newline + space * indent * level + "}"
    elif o is None:
        ret += "null"
    else:
        # NOTE: We added this check here to catch any Python objects that
        # didn't get turned into dictionaries.
        # This is to circumvent Swagger's inability to generate to_dicts that support
        # multi-dimensional dictionaries. This becomes an issue for us when we're storing
        # an array of metadata arrays
        if hasattr(o, "to_dict"):
            temp = del_nulls(o.to_dict())
            ret += dictionary_to_gromet_json(
                temp, fold_level, indent, level, parent_key
            )
        else:
            ret += str(o)
    return ret


def del_nulls(d):
    """
    This is just to make the json more human readable if wanted
    """
    for key, value in list(d.items()):
        if isinstance(value, list):
            for elem in value:
                if isinstance(elem, dict):
                    del_nulls(elem)
        if isinstance(value, dict):
            del_nulls(value)
        if value is None:
            del d[key]

    return d

if __name__ == "__main__":
    directory = "./dataset/test_code"
    dataset = preprocess_tokenized(directory)
    print(dataset)
    tree_dataset = preprocess_tree(directory)
    print(tree_dataset)