# 品質チェック
import random
import os
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model_name = "models/gemini-2.5-pro-exp-03-25"

generation_config = {
    "temperature": 0.0
}

model = genai.GenerativeModel(
    model_name=model_name,
    generation_config=generation_config,
    safety_settings=None,
)

def verify_ocr_quality(image_dir, extracted_text, quality_check_file):
    """特定のページに対する OCR 結果の品質を確認する"""
    # imageディレクトリからランダムにサンプルを２つ取ってくる
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(".jpg")]
    selected_image_paths = random.sample(image_paths, 2)  # 2つの画像をランダムに選択
    
    responses = []
    for image_path in selected_image_paths:
        image = Image.open(image_path)
        print(f"現在の処理: {image_path}")
        prompt = f"""
        私は OCR を使用して抽出されたテキストを持つドキュメントページがあります。
        
        元の画像と抽出されたテキストを比較し、エラーや省略を特定します。
        注意すべき点：
        1. 欠落しているテキスト
        2. 正しく認識されていない文字
        3. テーブル構造の問題
        4. 特殊文字や記号の問題
        
        抽出されたテキスト：
        {extracted_text}

        # 出力形式：
        ===
        Image: {image_path}\nAI Response: \n---
        """
        
        response = model.generate_content([prompt, image, image_path])
        print(f"LLMからの品質回答: {response}")
        responses.append(response)

    with open(quality_check_file, "w", encoding='utf-8') as f:
        for response in responses:
            f.write(response.text)

if __name__ == "__main__":
    # コマンドライン引数からパラメータを取得
    import sys
    
    if len(sys.argv) != 4:
        print("使用方法: python check_quality.py <image_dir> <extracted_text> <quality_check_file>")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    extracted_text = sys.argv[2]
    quality_check_file = sys.argv[3]
    print("品質チェックを開始します...")
    verify_ocr_quality(image_dir, extracted_text, quality_check_file)
    print("品質チェックが完了しました。")
