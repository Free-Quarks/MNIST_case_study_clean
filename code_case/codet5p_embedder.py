from transformers import AutoModel, AutoTokenizer
import subprocess


def embedder(code_string):
    """
    This function takes in a code string and will output the embedding vector for these sequence based on salesforce's codet5+ model.
    This is done by first encoding / tokenizing the input sequence and the running the encoded sequence through their trained embedding model. This is a 110m parameters embedding model that spans several programming languages including python. 

    Args:
        code_string (string): the string of code the get an embedding vector from

    Returns:
        embedding (torch.tensor): a torch tensor the of the 256 dimensional embedding
    """
    checkpoint = "Salesforce/codet5p-110m-embedding"
    device = "cuda:0"  # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    inputs = tokenizer.encode("def print_hello_world():\tprint('Hello World!')", return_tensors="pt").to(device)
    
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)
    embedding = model(inputs)

    return embedding

if __name__ == "__main__":

    checkpoint = "Salesforce/codet5p-110m-embedding"
    device = "cuda:0"  # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

    inputs = tokenizer.encode("def print_hello_world():\tprint('Hello World!')", return_tensors="pt").to(device)
    print(inputs.size()[1])
    embedding = model(inputs)
    print(embedding.size())
    subprocess.call(["nvidia-smi"]) # to profile memory usage, currently uses about 1 gb 
    print(f'Dimension of the embedding: {embedding.size()[0]}, with norm={embedding.norm().item()}')