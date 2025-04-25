import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# すでに前処理が終わっている前提
# テキストを分割
with open('ufj_sample.txt', encoding="utf-8") as f:
    text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 24,
    length_function = len,
    add_start_index = True,
)

texts = text_splitter.create_documents([text])

# チャンクのベクトル化
# 今回はAPIではなく、モデルを使ってベクトル化している
# HuggingFaceのembedding modelの設定
# GPUが利用できる前提
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large-instruct",  # ←フル ID を指定
    model_kwargs={"device": "cuda:0"},                     # CUDA 環境なら "cuda:0"
    encode_kwargs={"normalize_embeddings": False}
)

# FAISSへの格納とRetrieverの構築
# ページの内容を取得
texts_content = [doc.page_content for doc in texts]

# 文字列をベクトルに変換して、FAISSのインデックスを作成
db = FAISS.from_texts(texts_content, embeddings)
db.save_local("estate_playground.db")

db = FAISS.load_local("estate_playground.db", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

def setup_chatbot(retriever):
    """
    Azure OpenAIとFAISSベクターストアを使用してチャットボットをセットアップします。

    Returns:
        Callable: チャットボットの質問応答チェーンを実行するための関数。
    """

    llm = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("ENDPOINT_AZURE_OPENAI"),
            deployment_name="gpt-4o"
    )


    prompt_template = """システム: あなたは与えられたコンテキストに基づいて質問に答えるアシスタントです。回答は400文字以内とし、絶対に回答本文以外のテキスト（思考過程、前置き、後書きなど）を含めず、'回答:' から始めてください。

    コンテキスト:
    {context}

    質問:
    {question}"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain


# チャットボットの実行
if __name__ == "__main__":
    # コマンドライン引数から質問を取得
    question = "オフィスストックが拡大した地区では、賃料にどのような傾向が見られますか？"
    # question = " ".join(sys.argv[1:])

    chatbot = setup_chatbot(retriever)
    response = chatbot.invoke(question)
    print(response)