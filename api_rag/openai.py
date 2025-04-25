import os
import sys  # コマンドライン引数を扱うために必要

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()

# コマンドライン引数を取得


def setup_openai():
    