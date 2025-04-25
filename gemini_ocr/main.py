import os
import google.generativeai as genai
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import json
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# モデルのインスタンス作成
generation_config = {
    "temperature": 0.0
}

# モデル名を推奨されたプレビューバージョンに変更
model_name = "models/gemini-2.5-pro-preview-03-25"

# 必要であればsafetyを設定（今回はなし）
model = genai.GenerativeModel(
    model_name=model_name,
    generation_config=generation_config,
    safety_settings=None,
)

# テキスト位置検出用のシステム指示
text_detection_system_instructions = """
    テキストブロックのバウンディングボックスをJSONアレイとして返してください。コードフェンスやマスクは含めないでください。25個までのテキストブロックに制限します。
    各テキストブロックには 'label' フィールドにそのテキストの内容を含め、'box_2d' フィールドに位置情報を含めてください。
    位置情報は [y1, x1, y2, x2] の形式で、座標は1000で正規化されています（0から1000の範囲）。
"""

def parse_json(json_output):
    """JSONの出力からマークダウンフェンシングを削除する"""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # "```json"の前のすべてを削除
            json_output = json_output.split("```")[0]  # 閉じる"```"後のすべてを削除
            break
    return json_output

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

def detect_text_boxes(image_paths):
    """ドキュメント内のテキストのバウンディングボックスを検出する"""
    images = [Image.open(path) for path in image_paths]
    
    # システムインストラクションをプロンプトに組み込む
    prompt = f"""
    {text_detection_system_instructions}
    
    ドキュメント内のすべてのテキストブロックを検出し、各ブロックのテキスト内容とその位置を返してください。
    """
    
    response = model.generate_content(
        prompt,
        *images,
        generation_config={"temperature": 0.2},
    )
    
    # JSONをパース
    try:
        text_boxes = json.loads(parse_json(response.text))
        return text_boxes
    except json.JSONDecodeError as e:
        print(f"JSONデコードエラー: {e}")
        print(f"受信したテキスト: {response.text}")
        return []

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
    
    # 入力形式とパラメータを修正
    prompt = f"""
    {instruction}
    
    pytesseractによる事前抽出テキスト:
    {pytesseract_text}
    
    上記の事前抽出テキストを参考にして、正確なテキスト抽出を行ってください。
    """
    
    # APIコールを修正して安全にする
    try:
        response = model.generate_content(prompt, *images)
        extracted_text = response.text
        
        # テキストボックスを検出
        text_boxes = detect_text_boxes(image_paths)
        
        return {
            "extracted_text": extracted_text,
            "text_boxes": text_boxes
        }
    except Exception as e:
        print(f"API呼び出しエラー: {e}")
        return {
            "extracted_text": pytesseract_text,
            "text_boxes": []
        }

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
def process_large_pdf(pdf_path, output_folder, output_file, output_boxes_file):
    # 画像変換
    image_paths = pdf_to_images(pdf_path, output_folder)

    # 画像を意味単位で作成
    batches = batch_pdf_to_images(image_paths, 10)
    print(f"バッチ処理を実行します...")
    full_text = ""  # 変数を初期化
    all_text_boxes = []
    
    for i, batch in enumerate(batches):
        print(f"現在の処理中のバッチ：{i+1}")
        special_instruction = "すべてのテキストを抽出し、ドキュメント構造を維持"
        result = ocr_with_gemini(batch, special_instruction)
        batch_text = result["extracted_text"]
        text_boxes = result["text_boxes"]
        
        # ページ番号を追加
        for box in text_boxes:
            box["page"] = i + 1
        
        all_text_boxes.extend(text_boxes)
        full_text += f"\n\n--- バッチ {i+1} ---\n\n{batch_text}"
    
    # 全テキストの保存
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(full_text)
        
    # テキストボックスの保存
    with open(output_boxes_file, "w", encoding='utf-8') as f:
        json.dump(all_text_boxes, f, ensure_ascii=False, indent=2)

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

def visualize_text_boxes(image_path, text_boxes, output_path):
    """テキストボックスを視覚化する"""
    # 画像を開く
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # 画像のサイズを取得
    img_width, img_height = image.size
    
    # ランダムな色を生成する関数
    def random_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # フォントの設定（利用可能なフォントを使用）
    try:
        font = ImageFont.truetype("Arial.ttf", 15)
    except IOError:
        try:
            # Macでは代替フォント
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
        except IOError:
            # それでも失敗した場合はデフォルトフォント
            font = ImageFont.load_default()
    
    # 各テキストボックスを描画
    for box in text_boxes:
        # バウンディングボックスの座標を取得（正規化座標を実際の画像サイズに変換）
        if "box_2d" in box:
            y1, x1, y2, x2 = box["box_2d"]
            y1 = int(y1 * img_height / 1000)
            x1 = int(x1 * img_width / 1000)
            y2 = int(y2 * img_height / 1000)
            x2 = int(x2 * img_width / 1000)
            
            # ランダムな色でボックスを描画
            color = random_color()
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # テキストラベルを描画
            if "label" in box:
                text = box["label"]
                # PIL 9.0.0以降ではtextbboxを使用
                try:
                    # 新しいバージョン
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                except AttributeError:
                    # 古いバージョン向け
                    text_width, text_height = draw.textsize(text, font=font)
                
                # テキストの背景を描画
                draw.rectangle([x1, y1-text_height-4, x1+text_width+4, y1], fill=color)
                # テキストを描画
                draw.text((x1+2, y1-text_height-2), text, fill="white", font=font)
    
    # 視覚化した画像を保存
    image.save(output_path)
    print(f"視覚化された画像を保存しました: {output_path}")
    return image

def visualize_all_pages(image_folder, text_boxes_file, output_folder):
    """すべてのページのテキストボックスを視覚化する"""
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # テキストボックス情報を読み込む
    with open(text_boxes_file, 'r', encoding='utf-8') as f:
        all_text_boxes = json.load(f)
        
    # ページごとにテキストボックスをグループ化
    pages_boxes = {}
    for box in all_text_boxes:
        if "page" in box:
            page = box["page"]
            if page not in pages_boxes:
                pages_boxes[page] = []
            pages_boxes[page].append(box)
            
    # 各ページの画像に対して視覚化を実行
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    
    for i, image_file in enumerate(image_files):
        page = i + 1  # ページ番号（1から始まる）
        image_path = os.path.join(image_folder, image_file)
        output_path = os.path.join(output_folder, f"visualized_page_{page}.jpg")
        
        # このページのテキストボックスを取得
        if page in pages_boxes:
            page_boxes = pages_boxes[page]
            # 視覚化を実行
            visualize_text_boxes(image_path, page_boxes, output_path)
        else:
            print(f"ページ {page} のテキストボックス情報が見つかりません")
    
    print(f"すべてのページの視覚化が完了しました。結果は {output_folder} に保存されました。")

def main():
    pdf_path = "./data/raw_pdf/luxscape.pdf" # 処理するPDFのパス
    output_folder = "./data/output_images" # 画像を保存するディレクトリ
    output_file = "./data/content/luxscape.txt" # 抽出したテキストを保存するファイル
    output_file_normalized = "./data/content/luxscape_normalized.txt" # 正規化したテキストを保存するファイル
    output_boxes_file = "./data/content/luxscape_text_boxes.json" # テキストの位置情報を保存するファイル
    visualization_folder = "./data/visualization" # 視覚化結果を保存するディレクトリ

    # ディレクトリが存在しない場合は作成
    for folder in [os.path.dirname(output_file), visualization_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # pdfを画像へ変換
    image_paths = pdf_to_images(pdf_path, output_folder)
    print(f"画像のパス：{image_paths}")
    
    # 画像数が多い場合はprocess_large_pdfを使用
    if len(image_paths) > 3:  # 例えば3ページ以上の場合
        print(f"大きなページ数のPDFを検出しました。バッチ処理を実行します...")
        process_large_pdf(pdf_path, output_folder, output_file, output_boxes_file)
        
        # 保存したファイルを読み込んで正規化
        with open(output_file, "r", encoding='utf-8') as f:
            extracted_text = f.read()
    else:
        # 少ないページ数の場合は直接処理
        result = ocr_complex_document(image_paths)
        extracted_text = result["extracted_text"]
        text_boxes = result["text_boxes"]
        
        # テキストを保存
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(extracted_text)
            
        # テキストボックスを保存
        with open(output_boxes_file, "w", encoding='utf-8') as f:
            json.dump(text_boxes, f, ensure_ascii=False, indent=2)

    # テキストの正規化
    normalized_text = normalize_doc(extracted_text)

    # 保存
    with open(output_file_normalized, "w", encoding='utf-8') as f:
        f.write(normalized_text)

    # テキストボックスの視覚化
    visualize_all_pages(output_folder, output_boxes_file, visualization_folder)

    print(f"😆処理が完了しました🎉\nテキストは {output_file} & {output_file_normalized}に保存されました。")
    print(f"テキストボックスの位置情報は {output_boxes_file}に保存されました。")
    print(f"視覚化された結果は {visualization_folder}に保存されました。")

if __name__ == "__main__":
    main()