import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# モデルのインスタンス作成
generation_config = {
    "temperature": 0.0
}

model_name = "models/gemini-2.5-pro-exp-03-25"

# 必要であればsafetyを設定（今回はなし）
model = genai.GenerativeModel(
    model_name=model_name,
    generation_config=generation_config,
    safety_settings=None,
)

# スキーマを読み込む
with open("./schema/sample_schema.json", "r") as f:
    schema = json.load(f)

# スキーマをテキスト形式に変換
schema_text = json.dumps(schema, indent=2)

instruction = """
与えられたテキストをスキーマに沿って正規化してください。

注意すべき点：
- テキストは、不動産の物件概要書です。
- 必ずスキーマの指示に従って正規化してください。
- 情報がない場合は、nullを返してください。
- 最終的な出力は、スキーマに沿ったJSON形式のみを返してください。
"""

with open("./data/content/luxscape.txt", "r") as f:
    context = f.read()

# テキスト形式で渡す
prompt = f"""
## JSONスキーマ
{schema_text}

## 指示
{instruction}
"""

# スキーマと指示とコンテキストをモデルに渡す
response = model.generate_content(prompt + context)

# 出力をファイルに保存
with open("./data/content/luxscape_normalized.json", "w") as f:
    f.write(response.text)




