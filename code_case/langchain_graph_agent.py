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

# config
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
DEVICE = "cuda"