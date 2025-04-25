import os
import torch
import bitsandbytes as bnb

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline

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

# ★★★ 量子化設定を BitsAndBytesConfig で定義 ★★★
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

try:
    # モデルの準備！今回はdeepseek
    model = AutoModelForCausalLM.from_pretrained(
        "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese",
        quantization_config=quantization_config,
        # ★★★ device_map を明示的に指定 ★★★
        # '' はモデル全体を意味し、0 は最初のGPU (cuda:0) を意味する
        device_map={"": 0}
        # または単純に device_map="cuda:0" でも良い場合がある
        # device_map="cuda:0"
    )
    print("モデルのロードと量子化に成功しました。")

except ValueError as e:
    print(f"ValueErrorが発生しました: {e}")
    print("GPUメモリが本当に不足している可能性があります。nvidia-smiで確認してください。")
except Exception as e:
    print(f"モデルロード中に予期せぬエラーが発生しました: {e}")

tokenizer = AutoTokenizer.from_pretrained(
    "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese",
    )
streamer = TextStreamer(
    tokenizer, skip_prompt=True, skip_special_tokens=True
)

# ★★★ ChatML形式のプロンプトテンプレートを使用 ★★★
template_chatml = """<|im_start|>system
あなたは与えられたコンテキストに基づいて質問に答える優秀なアシスタントです。
回答は400文字以内とし、絶対に回答本文以外のテキスト（思考過程、前置き、後書きなど）を含めないでください。**必ず日本語で回答してください。**<|im_end|>
<|im_start|>user
以下のコンテキストを読んで、質問に**端的に**答えてください。

コンテキスト:
{context}

質問:
{question}<|im_end|>
<|im_start|>assistant
回答:
"""

prompt_chatml = PromptTemplate(
    template=template_chatml,
    input_variables=["context", "question"],
    template_format="f-string",
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512, # 512程度で十分なはず
    do_sample=True,
    temperature=0.0001,
    repetition_penalty=2.0,
    return_full_text=False
)
print("パイプラインの準備が完了しました。")

# RetrievalQAチェーンを再作成 (ChatMLプロンプトを使用)
qa = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(pipeline=pipe),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_chatml},
    verbose=True
)

query = "従業者数が増加したエリアの背景には何がありますか？"
print(f"\nQuery: {query}")
print("\nRunning RetrievalQA chain with ChatML prompt...")
answer = qa.invoke(query)

print("\n--- RetrievalQA Result ---")
print(f"Result: {answer['result']}")

# 後処理
cleaned_result = answer['result'].replace("<think>", "").replace("</think>", "").strip()
print(f"\n--- Cleaned Result ---")
print(cleaned_result)

with open('ufj_sample_answer_chatml.txt', 'w', encoding="utf-8") as f:
    f.write(cleaned_result)