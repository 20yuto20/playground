import os
import json
import io
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont, ImageColor
import pytesseract
import re

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

model_name = "models/gemini-2.5-pro-preview-03-25"

# 必要であればsafetyを設定（今回はなし）
# model = client.GenerativeModel(
#     model_name,
#     generation_config=types.GenerationConfig(temperature=0.2)
# )

# テキスト位置検出用のシステム指示
text_detection_system_instructions = """
    全てのテキストブロックのバウンディングボックスをJSONアレイとして返してください。コードフェンスやマスクは含めないでください。
    各テキストブロックには 'label' フィールドにそのテキストの内容を含め、'box_2d' フィールドに位置情報を含めてください。隣のテキストブロックとは","で必ず区切ってください。
    位置情報は [y1, x1, y2, x2] の形式で、座標は1000で正規化されています（0から1000の範囲）。
"""


ocr_extraction_system_instructions = """
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

def parse_json(json_output):
    """JSONの出力からマークダウンフェンシングを削除する"""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # "```json"の前のすべてを削除
            json_output = json_output.split("```")[0]  # 閉じる"```"後のすべてを削除
            break
    
    # JSONをパースしてみる
    try:
        # 有効なJSONかどうか確認
        parsed_json = json.loads(json_output)
        return json_output
    except json.JSONDecodeError as e:
        print(f"JSON解析エラー: {e}。修正を試みます...")
        
        # よくあるJSON形式エラーを修正
        # 余分なカンマを削除
        json_output = json_output.replace(",]", "]").replace(",}", "}")
        
        # 再度試行
        try:
            parsed_json = json.loads(json_output)
            return json_output
        except json.JSONDecodeError:
            print("JSON修正に失敗しました。空の配列を返します。")
            return "[]"

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

def extract_page_number(image_path):
    """画像ファイルパスからページ番号を抽出する"""
    # ファイル名のパターン：test_page_X.jpg からXを抽出
    match = re.search(r'test_page_(\d+)\.jpg', image_path)
    if match:
        return int(match.group(1))
    # ファイル名からページ番号を抽出できない場合はファイルパスの順序で推測
    return None

def detect_text_boxes(image_paths):
    """ドキュメント内のテキストのバウンディングボックスを検出する"""
    all_text_boxes = []
    
    # ファイル名とページ番号の対応マップを作成
    page_number_map = {}
    for i, path in enumerate(image_paths):
        page_num = extract_page_number(path)
        if page_num is None:
            page_num = i + 1  # ファイル名から抽出できなければデフォルトで順番を使用
        page_number_map[path] = page_num
    
    # 各ページごとに処理
    for image_path in image_paths:
        image = Image.open(image_path)
        page_num = page_number_map[image_path]
        
        prompt = "この画像内のすべてのテキストブロックを検出し、各ブロックのテキスト内容とその位置を返してください。"
        
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=text_detection_system_instructions,
            ),
        )
        
        # JSONをパース
        try:
            parsed_json = parse_json(response.text)
            text_boxes = json.loads(parsed_json)
            # 検出されたボックスに正しいページ番号を設定
            for box in text_boxes:
                box["page"] = page_num
            all_text_boxes.extend(text_boxes)
            print(f"ページ{page_num}から{len(text_boxes)}個のテキストボックスを検出しました")
        except json.JSONDecodeError as e:
            print(f"ページ{page_num}のJSONデコードエラー: {e}")
            print(f"受信したテキスト: {response.text}")
    
    return all_text_boxes

def ocr_with_gemini(image_paths):
    """geminiでの画像処理（pytesseractの結果も利用）"""
    print(f"画像のパス：{image_paths}")
    images = [Image.open(path) for path in image_paths]
    print(images)
    
    # ファイル名からページ番号を抽出
    page_numbers = {}
    for path in image_paths:
        page_num = extract_page_number(path)
        if page_num is None:
            # ファイル名から抽出できない場合はリストの順序を使用
            page_num = image_paths.index(path) + 1
        page_numbers[path] = page_num
    
    # pytesseractを使用して事前にテキストを抽出
    pytesseract_results = []
    for image in images:
        text = extract_text_with_pytesseract(image)
        pytesseract_results.append(text)
    
    # 抽出したテキストをプロンプトに含める
    pytesseract_text = "\n\n".join(pytesseract_results)
    
    # 通常のテキスト抽出
    prompt = f"""
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

    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, *images],           # put prompt + images in one list
        config=types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=ocr_extraction_system_instructions,
        ),
    )
    
    print(f"抽出されたテキスト：{response}")
    
    # テキストのバウンディングボックスも検出
    text_boxes = detect_text_boxes(image_paths)
    
    return {
        "extracted_text": response.text,
        "text_boxes": text_boxes
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
    
    return ocr_with_gemini(image_paths)

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
        
        # バッチ内の各ファイルからページ番号を抽出
        batch_page_info = []
        for path in batch:
            page_num = extract_page_number(path)
            if page_num is None:
                # ファイル名から抽出できない場合はリストの順序を使用
                page_num = image_paths.index(path) + 1
            batch_page_info.append(f"ページ{page_num}")
        
        batch_page_str = "、".join(batch_page_info)
        print(f"処理中のページ：{batch_page_str}")
        
        special_instruction = "すべてのテキストを抽出し、ドキュメント構造を維持"
        result = ocr_complex_document(batch)
        batch_text = result["extracted_text"]
        text_boxes = result["text_boxes"]
        
        all_text_boxes.extend(text_boxes)
        full_text += f"\n\n--- バッチ {i+1} ({batch_page_str}) ---\n\n{batch_text}"
    
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

    # ここでテキスト正規化用の専用インストラクションを使用
    normalize_instruction = """
    テキストの正規化と整形を行います。バッチ処理マーカーを削除し、文書構造を統一してください。
    テーブル形式が壊れている場合は修復し、段落の連続性を保ってください。
    JSONではなく、整形されたテキストとして出力してください。
    """

    response = client.models.generate_content(
        model=model_name,
        contents=[prompt + extracted_text],
        config=types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=normalize_instruction,
        ),
    )
    print(f"正規化プロセスの回答：{response}")
    return response.text

def visualize_text_boxes(image_path, text_boxes, output_path=None, page_num=None):
    """検出されたテキストボックスを視覚化する"""
    # 画像の読み込み
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # フォントを設定（使用可能なフォントがない場合はテキストのみ表示）
    try:
        # 日本語フォントがあればそれを使用
        font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 15) # OSのフォントを使用するため自身でフォントを指定してください！
    except IOError:
        font = None
    
    # カラーパレットの定義（複数の色を順番に使用）
    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
        "#FFA500", "#800080", "#008000", "#800000", "#008080", "#000080"
    ]
    
    # ページ番号が指定されている場合は、そのページのボックスのみをフィルタリング
    if page_num is not None:
        filtered_boxes = [box for box in text_boxes if box.get("page", 1) == page_num]
    else:
        filtered_boxes = text_boxes
    
    # 各テキストボックスを描画
    for i, box in enumerate(filtered_boxes):
        # box_2dのフォーマットは [y1, x1, y2, x2]
        if "box_2d" in box:
            # 正規化された座標を実際の画像サイズに変換
            y1, x1, y2, x2 = box["box_2d"]
            
            # 座標を0-1000の範囲から画像サイズに合わせて変換
            img_width, img_height = image.size
            x1 = int(x1 * img_width / 1000)
            y1 = int(y1 * img_height / 1000)
            x2 = int(x2 * img_width / 1000)
            y2 = int(y2 * img_height / 1000)
            
            # 色の選択（循環して使用）
            color = colors[i % len(colors)]
            
            # 長方形を描画
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # テキストラベルを描画（可能な場合）
            label = box.get("label", "")
            if label and font:
                # 現在のPILバージョンに合わせたテキストサイズ取得方法
                try:
                    # PILの新しいバージョン
                    text_bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                except AttributeError:
                    # 古いPILバージョン向け互換性維持
                    try:
                        text_width, text_height = draw.textsize(label, font=font)
                    except:
                        # サイズ取得に失敗した場合のフォールバック
                        text_width, text_height = 100, 15
                
                # 背景付きでテキストを描画
                draw.rectangle([x1, y1 - text_height - 2, x1 + text_width, y1], fill=color)
                draw.text((x1, y1 - text_height - 2), label, fill="white", font=font)
            elif label:
                # フォントがない場合は背景なしでテキストを描画
                draw.text((x1, y1 - 15), label, fill=color)
    
    # 結果を保存または表示
    if output_path:
        image.save(output_path)
        print(f"✅視覚化画像を保存しました: {output_path}")
    
    return image

# main関数を拡張して視覚化機能を追加
def main():
    pdf_path = "./data/raw_pdf/C＆Mコーポレーション.pdf" # 処理するPDFのパス
    output_folder = "./data/output_images" # 画像を保存するディレクトリ
    output_file = "./data/content/C＆Mコーポレーション.txt" # 抽出したテキストを保存するファイル
    output_file_normalized = "./data/content/C＆Mコーポレーション_normalized.txt" # 正規化したテキストを保存するファイル
    output_boxes_file = "./data/content/9GATES_text_boxes.json" # テキストの位置情報を保存するファイル
    visualized_output_folder = "./data/visualized_images" # 視覚化画像を保存するディレクトリ

    # 視覚化出力用ディレクトリの作成
    if not os.path.exists(visualized_output_folder):
        os.makedirs(visualized_output_folder)

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
        
        # 位置情報を読み込み
        with open(output_boxes_file, "r", encoding='utf-8') as f:
            all_text_boxes = json.load(f)
        
        # 各ページのテキストボックスを視覚化
        page_boxes = {}
        for box in all_text_boxes:
            page = box.get("page", 1)
            if page not in page_boxes:
                page_boxes[page] = []
            page_boxes[page].append(box)
        
        # ページごとに視覚化
        for page, boxes in page_boxes.items():
            if page <= len(image_paths):
                image_path = image_paths[page-1]
                output_viz_path = os.path.join(visualized_output_folder, f"visualized_page_{page}.jpg")
                visualize_text_boxes(image_path, boxes, output_viz_path, page)
    else:
        # 少ないページ数の場合は直接処理
        
        # テキストボックスをページごとに検出
        text_boxes = detect_text_boxes(image_paths)
        
        # OCRを使用してテキストを抽出
        result = ocr_complex_document(image_paths)
        extracted_text = result["extracted_text"]
        
        # テキストを保存
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(extracted_text)
            
        # テキストボックスを保存
        with open(output_boxes_file, "w", encoding='utf-8') as f:
            json.dump(text_boxes, f, ensure_ascii=False, indent=2)
        
        # 各ページを視覚化
        for i, image_path in enumerate(image_paths):
            page_num = i + 1
            output_viz_path = os.path.join(visualized_output_folder, f"visualized_page_{page_num}.jpg")
            visualize_text_boxes(image_path, text_boxes, output_viz_path, page_num)

    # テキストの正規化
    normalized_text = normalize_doc(extracted_text)

    # 保存
    with open(output_file_normalized, "w", encoding='utf-8') as f:
        f.write(normalized_text)

    print(f"😆処理が完了しました🎉\nテキストは {output_file} & {output_file_normalized}に保存されました。")
    print(f"テキストボックスの位置情報は {output_boxes_file}に保存されました。")
    print(f"視覚化された画像は {visualized_output_folder}に保存されました。")

if __name__ == "__main__":
    main()