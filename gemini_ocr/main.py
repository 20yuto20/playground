import os
import json
import io
import google.generativeai as genai
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont, ImageColor
import pytesseract

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

def parse_json(json_output: str):
    """座標がJSON形式で出力されるので、それを変換"""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

def boundig_box_instructions():
    """テキストボックスの座標を取得するためのプロンプト"""
    bounding_box_instructions = """
    Return bounding boxes as a JSON array with labels. Never return masks or code fencing.
    If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
    """
    return bounding_box_instructions

def plot_bounding_boxes(image_path, bounding_boxes):
    """テキスト情報にバウンディングボックスを描画する"""

def pdf_to_images(pdf_path, output_folder, dpi=600):
    # ディレクトリ作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # PDFから画像に変換（高解像度設定）
    images = convert_from_path(pdf_path, dpi=dpi)

    # 保存前に画像処理を追加
    image_paths = []
    for i, image in enumerate(images):
        # コントラスト強化と二値化処理を追加
        enhanced_image = enhance_image_for_ocr(image)
        image_path = os.path.join(output_folder, f"test_page_{i+1}.jpg")
        enhanced_image.save(image_path, "JPEG", quality=95)  # 高品質で保存
        image_paths.append(image_path)
    
    return image_paths

def enhance_image_for_ocr(image):
    """数値認識に特化した画像処理を行う"""
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance

    # 画像をnumpy配列に変換
    image_array = np.array(image)
    
    # グレースケール変換
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # ノイズ除去
    image_array = cv2.GaussianBlur(image_array, (5, 5), 0)

    # 2値化
    _, image_array = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # コントラスト強調
    image_array = cv2.convertScaleAbs(image_array, alpha=1.5, beta=0)

    # numpy配列をPIL画像に変換
    image = Image.fromarray(image_array)
    
    return image

# 大規模なPDFのバッチ処理
def batch_pdf_to_images(image_paths, batch_size=10):  #　バッチサイズは適宜調整！
    """画像をバッチにグループ化"""
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        yield batch

def extract_text_with_pytesseract(image):
    """pytesseractを使用してテキストを抽出する"""
    # 日本語OCRの設定（必要に応じて）
    custom_config = r'--oem 3 --psm 6 -l jpn'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

def ocr_with_gemini(image_paths, instruction):
    """geminiでの画像処理（pytesseractの結果も利用）"""
    print(f"画像のパス：{image_paths}")
    images = [Image.open(path) for path in image_paths]
    print(images)
    
    # pytesseractを使用して事前にテキストを抽出
    pytesseract_results = []
    for image in images:
        text = extract_text_with_pytesseract(image)
        pytesseract_results.append(text)
    
    # 抽出したテキストをプロンプトに含める
    pytesseract_text = "\n\n".join(pytesseract_results)
    
    prompt = f"""
    {instruction}

    これは PDF ドキュメントのページです。構造を維持しながら、すべてのテキストコンテンツを抽出してください。
    テーブル、列、見出し、および構造化されたコンテンツに特に注意を払ってください。
    段落の区切りと書式を維持してください。
    
    別のOCRエンジン(pytesseract)から抽出したテキストも参考にしてください：
    ---
    {pytesseract_text}
    ---
    
    この参考テキストにはエラーが含まれている可能性がありますが、数字や表構造の認識に役立つかもしれません。
    最終的な出力は、画像の内容を正確に反映し、適切にフォーマットされたものにしてください。
    """

    response = model.generate_content([prompt, *images, instruction])
    # テキストが長すぎてエラーが出た時は、process_large_pdfを実行
    print(f"抽出されたテキスト：{response}")

    return response.text

# 複雑なドキュメントへの対応（不動産レポートのグラフなど）
def ocr_complex_document(image_paths):
    instruction = """
    これらのドキュメントページからすべてのテキストコンテンツを抽出してください。

    テーブルの場合：
    1. マークダウンテーブル形式を使用してテーブル構造を維持
    2. すべての列ヘッダーと行ラベルを保持
    3. 数値データが正確にキャプチャされていることを確認

    マルチカラムレイアウトの場合：
    1. 左から右へ列を処理
    2. 異なる列のコンテンツを明確に区別

    チャートやグラフの場合：
    1. チャートのタイプを説明
    2. 可視の軸ラベル、凡例、データポイントを抽出
    3. タイトルやキャプションを抽出

    特に注意すべき点：
    1. すべての数値の正確な転記を確認（最優先事項）
    2. +, -, ±などの数値の前の記号
    3. ()の中の文字と数値
    4. テーブルや文中に存在する正式な物件名と関連金額

    すべての見出し、フッター、ページ番号、脚注を維持してください。
    """
    
    return ocr_with_gemini(image_paths, instruction)

# でっかいドキュメントを処理する
def process_large_pdf(pdf_path, output_folder, output_file):
    # 画像変換
    image_paths = pdf_to_images(pdf_path, output_folder)

    # 画像を意味単位で作成
    batches = batch_pdf_to_images(image_paths, 10)
    print(f"バッチ処理を実行します...")
    full_text = ""  # 変数を初期化
    for i, batch in enumerate(batches):
        print(f"現在の処理中のバッチ：{i+1}")
        special_instruction = "すべてのテキストを抽出し、ドキュメント構造を維持"
        batch_text = ocr_with_gemini(batch, special_instruction)
        full_text += f"\n\n--- バッチ {i+1} ---\n\n{batch_text}"
    
    # 全テキストの保存
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(full_text)

# テキスト抽出後の一貫性の確保
def normalize_doc(extracted_text):
    prompt = """
    以下のテキストは、大きな PDF ドキュメントからバッチで抽出されました。
        内容を調和させるために：
        1. すべてのバッチ分離マーカーを削除
        2. 一貫した書式を確保
        3. バッチ境界でのテーブル構造の問題を修正
        4. バッチ境界を越えた段落とセクションの流れが自然であることを確認
        
    元の抽出されたテキスト：
    """

    response = model.generate_content(prompt + extracted_text)
    print(f"正規化プロセスの回答：{response}")
    return response.text

def main():
    pdf_path = "./data/raw_pdf/luxscape.pdf" # 処理するPDFのパス
    output_folder = "./data/output_images" # 画像を保存するディレクトリ
    output_file = "./data/content/luxscape.txt" # 抽出したテキストを保存するファイル
    output_file_normalized = "./data/content/luxscape_normalized.txt" # 正規化したテキストを保存するファイル

    # pdfを画像へ変換
    image_paths = pdf_to_images(pdf_path, output_folder)
    print(f"画像のパス：{image_paths}")
    
    # 画像数が多い場合はprocess_large_pdfを使用
    if len(image_paths) > 3:  # 例えば3ページ以上の場合
        print(f"大きなページ数のPDFを検出しました。バッチ処理を実行します...")
        process_large_pdf(pdf_path, output_folder, output_file)
        
        # 保存したファイルを読み込んで正規化
        with open(output_file, "r", encoding='utf-8') as f:
            extracted_text = f.read()
    else:
        # 少ないページ数の場合は直接処理
        extracted_text = ocr_complex_document(image_paths)
        
        # 保存
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(extracted_text)

    # テキストの正規化
    normalized_text = normalize_doc(extracted_text)

    # 保存
    with open(output_file_normalized, "w", encoding='utf-8') as f:
        f.write(normalized_text)

    print(f"😆処理が完了しました🎉\nテキストは {output_file} & {output_file_normalized}に保存されました。")

if __name__ == "__main__":
    main()