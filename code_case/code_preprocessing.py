from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import requests
import json
import networkx as nx
from enum import Enum, auto

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

# Enum for the node types
class NodeType(Enum):
    MODULE = auto()
    FUNCTION = auto()
    PREDICATE = auto()
    LANGUAGE_PRIMITIVE = auto()
    ABSTRACT = auto()
    EXPRESSION = auto()
    LITERAL = auto()
    IMPORTED = auto()
    IMPORTED_METHOD = auto()
    TEMP = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return NodeType[s.upper()]
        except KeyError:
            raise ValueError(f"'{s}' is not a valid NodeType")

# Class for storing info about each node in the tree, idx is inherited from parent for atomic nodes
class NodeToken:
    def __init__(self):
        self.type = NodeType.TEMP # node type
        self.idx = 0 # idx in the fn_array, 0 if the source is in the top level part
        self.name = None # name if it has one (only functions)
        self.bf = False # boolean on if there is substructure
        self.tokens = None # tokens sliced from source code

    def __eq__(self, other):
        if isinstance(other, NodeToken):
            return self.type == other.type and self.bf == other.bf and self.idx == other.idx
        return False

    def __hash__(self):
        return hash((self.name, self.type, self.idx))
    
    def __repr__(self):
        return f"type: {self.type}, idx: {self.idx}"
    def __str__(self):
        return f"type: {self.type}, idx: {self.idx}"

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

def preprocess_tree(fn_directory, code_directory):
    """
    This function takes in the directory of the function networks to be processed into a dataset of code trees. Each node will have a subet of tokens and the feature vector for that node will come from the embedding model we are using. 

    Args:
        fn_directory (string): the string of the directory of function networks to be processed
        code_directory (string): the string of the directory of the source code, will be used to collect and embbed tokens
    Returns:
        dataset (Dataset): a dataset of processed code
    """
    # There are 3 different aspects of information we want to encode in each node:
    #       1) The embedding vector of the tokens in the node
    #       2)** The positional encodings using eigenvectors of the Laplacian of the graph or other options
    #       3) The node type encoding as well, function, literal, etc. 

    # 1) and 2) will be added together as is usual for the embedding and positional information in transformer architectures
    # 3) is still being thought about, but perhaps will be a seperate dimension of the feature space. 
    # ** 2) can only be consturcted after the entire graph has been constructed since it encodes relative locational information

    tree_list = []
    # read in the fn's
    for filename in os.listdir(fn_directory):
        
        G = nx.Graph() # initialize undirected graph

        # add top level module node
        mod = NodeToken()
        mod.type = NodeType.MODULE
        G.add_node(mod)

        # find and read in the matching source code file
        code_file_id = filename.split("-")[2].split(".")[0]
        for codename in os.listdir(code_directory):
            if codename.split("-")[2].split(".")[0] == code_file_id:
                c = os.path.join(code_directory, codename)
                with open(c, 'r', encoding='utf-8') as code_file:
                    code_data = code_file.read()
                    code_file.close()

        # load the fn as json dict
        f = os.path.join(fn_directory, filename)
        with open(f,'r') as fn_file:
            fn_data = json.load(fn_file)
            fn_file.close()
        

        #---- Constuct the tree ----#
        # This is done in 3 passes, one to get top executable level, one to get the top fn_array level, and 
        # one last pass to fill in subsructure and add edges to join referenced entries

        # 1st pass, construct first layer of tree for executable level
        module_bf = fn_data['modules'][0]['fn']['bf']
        for i, ent in enumerate(module_bf):
            t = NodeToken()
            t.type = NodeType.from_string(ent['function_type'])
            if 'name' in ent:
                t.name = ent['name']
            if 'metadata' in ent:
                t.tokens = get_tokens(code_data, fn_data, ent['metadata'])
            if 'body' in ent:
                t.bf = True
                t.idx = ent['body']
            G.add_node(t)
            G.add_edge(mod, t)



        # 2nd pass, construct first layer of tree for fn_array level
        fn_array = fn_data['modules'][0]['fn_array']
        for j, obj in enumerate(fn_array):
            t = NodeToken()
            t.type = NodeType.from_string(obj['b'][0]['function_type'])
            t.idx = j+1
            if 'bf' in obj:
                t.bf = True
            # now check to make sure it wasn't already created in executable pass
            novel_node = True
            for node in list(G):
                if node == t:
                    novel_node = False
                    break
            if novel_node:
                if 'name' in obj['b'][0]:
                    t.name = obj['b'][0]['name']
                if 'metadata' in obj['b'][0]:
                    t.tokens = get_tokens(code_data, fn_data, obj['b'][0]['metadata'])
                G.add_node(t)
                G.add_edge(mod, t)
        
        # 3rd pass now that we have all the top level objects it is time to add the substructure that 
        # exists in the expressions, predicates, and functions 
        for node in list(G):
            if node.bf:
                bf_list = fn_array[node.idx - 1]['bf']
                for ent in bf_list:
                    # construct node, check if novel, add node and edge
                    t = NodeToken()
                    t.type = NodeType.from_string(ent['function_type'])
                    if 'body' in ent:
                        t.idx = ent['body']
                        # if the node have substructure the parent has already been created, so we simply connect it
                        t.bf = True
                        # find the node and add edge once found
                        for old_node in list(G):
                            if old_node == t:
                                G.add_edge(node, old_node)
                                break
                    else:
                        t.idx = node.idx
                        if 'name' in ent:
                            t.name = ent['name']
                        if 'metadata' in ent:
                            t.tokens = get_tokens(code_data, fn_data, ent['metadata'])
                        G.add_node(t)
                        G.add_edge(node, t)
        
        tree_list.append(G)


    dataset = []
    return dataset

def get_tokens(code_data, fn_data, meta_data_idx):
    """
    This function takes in the code directory and the fn_file_name to match the code file to the fn file. This function
    then returns the tokens for the correcsponding line number given as well. 

    Args:
        code_data (string): The source code to slice from for tokens
        fn_data (string): The fn data to pull the line the splice from
        meta_data_idx (integer): The idx into the metadata which will contain the line number to splice the tokens with

    Return: 
        tokens: (string): The string of the tokens from the line to be embedded as a feature for the node. 
    """
    tokens = ""
    return tokens

def encode_graph(graph):
    """
    This function takes in the human-readable graph and will create a new graph where each node is encoded with our given feature choices. This function will also include the positional encoding in the graph, so it will be ready to be fed into GNN right away. 

    Args:
        graph (networkx graph): This is the input human readable graph with NodeToken objects as the nodes
    
    Return:
        encoded_graph (networkx graph): This is the encoded version of the human readable graph with positional encoding as well.
    """
    encode_graph = graph
    return encode_graph

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