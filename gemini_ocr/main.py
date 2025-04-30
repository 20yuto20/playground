import os
import json
import google.generativeai as genai
# from google.genai import types
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont, ImageColor

import pytesseract
import re
import glob
import datetime
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import cv2
import numpy as np
from functools import wraps
import imutils
import base64

load_dotenv()

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"⌛️処理時間 - {func.__name__}: {execution_time:.2f}秒")
        return result, execution_time
    return wrapper

class OCRProcessor:
    def __init__(self, cfg: DictConfig):
        """OCRプロセッサの初期化"""
        self.cfg = cfg
        
        # Gemini APIクライアントの設定
        genai.configure(api_key=os.getenv(self.cfg.model.api_key_env))
        
        # GenAIモデルのインスタンス化
        self.generation_config = {
            "temperature": cfg.model.temperature
        }
        
        # GenerativeModelインスタンスを作成
        self.model = genai.GenerativeModel(
            model_name=cfg.model.name,
            generation_config=self.generation_config,
            safety_settings=None,
        )
        
        # 処理時間追跡用辞書
        self.execution_times = {}
        
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.setup_output_dirs()
        
    def setup_client(self):
        """Gemini APIクライアントを設定"""
        load_dotenv()
        # 最新のクライアント初期化方法
        api_key = os.getenv(self.cfg.model.api_key_env)
        # 最新のSDKに対応した初期化方法
        self.client = genai.Client(api_key=api_key)
        
    def setup_output_dirs(self):
        """出力ディレクトリを設定"""
        base_dir = f"./data/results_{self.timestamp}"
        os.makedirs(base_dir, exist_ok=True)
        
        self.output_image_dir = os.path.join(base_dir, self.cfg.output.images_folder)
        self.output_content_dir = os.path.join(base_dir, self.cfg.output.content_folder)
        self.output_visualized_dir = os.path.join(base_dir, self.cfg.output.visualized_folder)
        
        os.makedirs(self.output_image_dir, exist_ok=True)
        os.makedirs(self.output_content_dir, exist_ok=True)
        os.makedirs(self.output_visualized_dir, exist_ok=True)
    
    @timer_decorator
    def detect_rotation_and_correct(self, image):
        """画像の向きを検出して修正"""
        if not self.cfg.image_processing.auto_rotate:
            return image

        # PILイメージをOpenCV形式に変換
        img_array = np.array(image)
        rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        try:
            # OCRエンジンのOSD機能を使用して向きを検出
            results = pytesseract.image_to_osd(rgb, output_type=pytesseract.Output.DICT)
            angle = results["rotate"]
            
            if abs(angle) > 0.1:  # 微小な角度は無視
                print(f"検出された回転角度: {angle}°")
                # imutilsを使用して画像を回転（境界を保持）
                rotated = imutils.rotate_bound(img_array, angle=angle)
                return Image.fromarray(rotated)
            else:
                return image
        except Exception as e:
            print(f"向き検出に失敗しました: {str(e)}")
            return image

    def detect_text_angle(self, img):
        """OCRを使用してテキスト角度を検出"""
        try:
            # グレースケール変換
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # コントラスト強調（CLAHE法）
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # 追加のコントラストストレッチング
            enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
            
            # 複数の閾値処理
            local_thresh_methods = [
                # 適応的閾値処理（異なるパラメータで）
                cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
                cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5),
                # 大津の二値化
                cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            ]
            
            text_angles = []
            for thresh in local_thresh_methods:
                try:
                    # 日本語テキスト検出を試行
                    osd = pytesseract.image_to_osd(thresh, config='--psm 0 -l jpn')
                    angle = int(osd.split('\n')[2].split(':')[1].strip())
                    text_angles.append(angle)
                except Exception as e:
                    print(f"日本語テキスト検出に失敗: {str(e)}")
                    try:
                        # デフォルトのテキスト検出を試行
                        osd = pytesseract.image_to_osd(thresh)
                        angle = int(osd.split('\n')[2].split(':')[1].strip())
                        text_angles.append(angle)
                    except Exception as e:
                        print(f"デフォルトテキスト検出に失敗: {str(e)}")
                        continue
            
            return text_angles
        except Exception as e:
            print(f"テキスト角度検出に失敗: {str(e)}")
            return []

    def correct_image_rotation(self, image, angle):
        """高品質な画像回転処理"""
        try:
            if abs(angle) < 0.1:  # 非常に小さな角度はスキップ
                return image
            
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            
            # クロップを防ぐための新しい寸法を計算
            angle_rad = np.abs(np.radians(angle))
            new_w = int(h * np.abs(np.sin(angle_rad)) + w * np.abs(np.cos(angle_rad)))
            new_h = int(h * np.abs(np.cos(angle_rad)) + w * np.abs(np.sin(angle_rad)))
            
            # クロップを防ぐための変換行列の調整
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            
            # 角度に基づいて異なる補間方法を使用
            if abs(angle) in [90, 180, 270]:
                # 直角にはNEARESTを使用（アーティファクト防止）
                interpolation = cv2.INTER_NEAREST
            else:
                # その他の角度にはCUBICを使用
                interpolation = cv2.INTER_CUBIC
            
            # ボーダーレプリケーションで回転を実行
            rotated = cv2.warpAffine(image, M, (new_w, new_h),
                                    flags=interpolation,
                                    borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
            
        except Exception as e:
            print(f"correct_image_rotation でエラー発生: {str(e)}")
            return image
        
    @timer_decorator
    def enhance_image_for_ocr(self, image):
        """数値認識に特化した画像処理を行う"""
        # 画像をnumpy配列に変換
        image_array = np.array(image)
        
        # グレースケール変換
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # ノイズ除去
        kernel_size = tuple(self.cfg.image_processing.gaussian_blur.kernel_size)
        sigma = self.cfg.image_processing.gaussian_blur.sigma
        image_array = cv2.GaussianBlur(image_array, kernel_size, sigma)

        # 2値化
        _, image_array = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # コントラスト強調
        alpha = self.cfg.image_processing.contrast.alpha
        beta = self.cfg.image_processing.contrast.beta
        image_array = cv2.convertScaleAbs(image_array, alpha=alpha, beta=beta)

        # numpy配列をPIL画像に変換
        image = Image.fromarray(image_array)
        
        return image
        
    @timer_decorator
    def pdf_to_images(self, pdf_path):
        """PDFを画像に変換して前処理"""
        # PDFから画像に変換（高解像度設定）
        images = convert_from_path(pdf_path, dpi=self.cfg.pdf.dpi)

        # 保存前に画像処理を追加
        image_paths = []
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        for i, image in enumerate(images):
            # 画像の向き検出・修正
            corrected_image, _ = self.detect_rotation_and_correct(image)
            
            # コントラスト強化と二値化処理
            enhanced_image, _ = self.enhance_image_for_ocr(corrected_image)
            
            image_path = os.path.join(self.output_image_dir, f"{pdf_name}_page_{i+1}.jpg")
            enhanced_image.save(image_path, "JPEG", quality=95)  # 高品質で保存
            image_paths.append(image_path)
        
        return image_paths
        
    @timer_decorator
    def extract_text_with_pytesseract(self, image):
        """pytesseractを使用してテキストを抽出する"""
        # 日本語OCRの設定
        text = pytesseract.image_to_string(image, config=self.cfg.ocr.pytesseract_config)
        return text
        
    @timer_decorator
    def extract_page_number(self, image_path):
        """画像ファイルパスからページ番号を抽出する"""
        match = re.search(r'page_(\d+)\.jpg', image_path)
        if match:
            return int(match.group(1))
        return None
        
    @timer_decorator
    def detect_text_boxes(self, image_paths):
        """ドキュメント内のテキストのバウンディングボックスを検出する (JSONモード強制)"""

        # --- System インストラクション ---
        text_detection_system_instructions = """
        全てのテキストブロックのバウンディングボックスをJSONアレイとして返してください。
        回答は strict な JSON のみとし、他の文字は一切含めないでください。
        各テキストブロックには "label" と "box_2d" を含め、
        "box_2d" は [y1, x1, y2, x2]（0-1000 正規化）とします。
        """

        # JSON専用の設定を作成
        json_config = self.generation_config.copy()
        json_config["response_mime_type"] = "application/json"
        
        # JSON用のモデルインスタンスを作成
        json_model = genai.GenerativeModel(
            model_name=self.cfg.model.name,
            generation_config=json_config,
            safety_settings=None,
        )

        all_text_boxes = []

        # ファイル名とページ番号の対応マップ
        page_number_map = {}
        for i, path in enumerate(image_paths):
            page_num_result = self.extract_page_number(path)
            page_num = page_num_result[0] if isinstance(page_num_result, tuple) else page_num_result
            if page_num is None:
                page_num = i + 1
            page_number_map[path] = page_num

        # 各ページごとに処理
        for image_path in image_paths:
            try:
                page_num = page_number_map[image_path]
                
                # 画像読み込み
                image = Image.open(image_path)
                
                # プロンプト作成
                prompt = f"""
                {text_detection_system_instructions}
                
                ---
                
                この画像内のすべてのテキストブロックを検出してください。
                """
                
                # GenerativeModelを使用してAPI呼び出し
                response = json_model.generate_content([prompt, image])
                
                try:
                    # 応答からJSONを抽出
                    text_boxes = json.loads(response.text)
                    for box in text_boxes:
                        box["page"] = page_num
                    all_text_boxes.extend(text_boxes)
                    print(f"ページ{page_num}から{len(text_boxes)}個のテキストボックスを検出しました")
                except json.JSONDecodeError as e:
                    print(f"ページ{page_num}のJSONデコードエラー: {e}")
                    print(f"受信したテキスト: \n {response.text}")

                    pdf_name = os.path.splitext(os.path.basename(image_path))[0]
                    # 失敗ケースをファイルに保存
                    with open(os.path.join(self.output_content_dir, f"failed_text_boxes_{pdf_name}.json"), "a") as f:
                        f.write(f"ページ{page_num}のJSONデコードエラー: {e}\n")
                        f.write(f"受信したテキスト: \n {response.text}\n\n")
                    continue
                
            except Exception as e:
                print(f"ページ処理中のエラー: {e}")
                continue

        return all_text_boxes

    @timer_decorator
    def parse_json(self, json_output):
        """JSONの出力からマークダウンフェンシングを削除し、修正する"""
        # マークダウンフェンスを処理
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json" or line.strip() == "```":
                json_output = "\n".join(lines[i+1:])  # "```json"の前のすべてを削除
                json_output = json_output.split("```")[0]  # 閉じる"```"後のすべてを削除
                break
        
        # よくある問題を事前修正
        try:
            # 有効なJSONかどうか確認
            parsed_json = json.loads(json_output)
            return json_output
        except json.JSONDecodeError as e:
            print(f"JSON解析エラー: {e}。修正を試みます...")
            
            # 段階的に修正を試みる
            # 1. 余分なカンマを削除
            json_output = json_output.replace(",]", "]").replace(",}", "}")
            
            # 2. シングルクォートをダブルクォートに置換
            json_output = json_output.replace("'", "\"")
            
            # 3. プロパティ名のクォートがない場合の修正を試みる
            import re
            json_output = re.sub(r'(\s*)(\w+)(\s*):([^/])', r'\1"\2"\3:\4', json_output)
            
            # 4. Pythonスタイルのコメントを削除
            json_output = re.sub(r'#.*?\n', '\n', json_output)
            
            # 5. 非標準のUnicode文字や制御文字を修正
            json_output = ''.join(c for c in json_output if c.isprintable() or c in ['\n', '\t', ' '])
            
            # 再度試行
            try:
                parsed_json = json.loads(json_output)
                print("JSON修正に成功しました！")
                return json_output
            except json.JSONDecodeError as e2:
                print(f"JSON修正に失敗しました: {e2}")
                print(f"問題のJSON: {json_output[:200]}...")  # 最初の200文字のみ表示
                return "[]"
                
    @timer_decorator
    def batch_pdf_to_images(self, image_paths):
        """画像をバッチにグループ化"""
        for i in range(0, len(image_paths), self.cfg.ocr.batch_size):
            batch = image_paths[i:i+self.cfg.ocr.batch_size]
            yield batch
                
    @timer_decorator
    def ocr_with_gemini(self, image_paths):
        """Gemini を用いた OCR"""
        # システム指示
        system_instruction = """
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

        # pytesseractのテキスト抽出結果も組み合わせる
        pytesseract_texts = []
        for path in image_paths:
            img = Image.open(path)
            txt, _ = self.extract_text_with_pytesseract(img)
            pytesseract_texts.append(txt)
        pytesseract_dump = "\n\n".join(pytesseract_texts)

        # プロンプトを作成
        prompt = f"""
        {system_instruction}
        
        これは PDF ドキュメントのページです。構造を維持しながら全文抽出してください。
        ---
        参考 OCR（pytesseract）結果：
        {pytesseract_dump}
        ---
        """

        # プロンプトと画像を含む入力作成
        input_content = [prompt]
        
        # 画像を追加
        for path in image_paths:
            img = Image.open(path)
            input_content.append(img)

        # GenerativeModelを使用してAPI呼び出し
        response = self.model.generate_content(input_content)

        # テキストボックス検出
        text_boxes_result, _ = self.detect_text_boxes(image_paths)
        text_boxes = text_boxes_result[0] if isinstance(text_boxes_result, tuple) else text_boxes_result

        return {
            "extracted_text": response.text,
            "text_boxes": text_boxes,
        }

    @timer_decorator
    def normalize_doc(self, extracted_text):
        """バッチ結合後テキストの整形"""
        # プロンプト作成
        prompt = f"""
        テキストの正規化と整形を行います。バッチ処理マーカーを削除し、文書構造を統一してください。
        テーブル形式が壊れている場合は修復し、段落の連続性を保ってください。
        JSONではなく、整形されたテキストとして出力してください。

        以下のテキストは、大きな PDF ドキュメントからバッチで抽出されました。
            内容を調和させるために：
            1. すべてのバッチ分離マーカーを削除
            2. 一貫した書式を確保
            3. バッチ境界でのテーブル構造の問題を修正
            4. バッチ境界を越えた段落とセクションの流れが自然であることを確認

        元の抽出されたテキスト：
        {extracted_text}
        """

        # GenerativeModelを使用してAPI呼び出し
        response = self.model.generate_content(prompt)

        return response.text
        
    @timer_decorator
    def ocr_complex_document(self, image_paths):
        """複雑なドキュメントへの対応"""
        ocr_result = self.ocr_with_gemini(image_paths)
        return ocr_result[0] if isinstance(ocr_result, tuple) else ocr_result  # タプルの場合は結果だけを返す
        
    @timer_decorator
    def process_large_pdf(self, pdf_path):
        """大規模PDFの処理"""
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_file = os.path.join(self.output_content_dir, f"{pdf_name}.txt")
        output_file_normalized = os.path.join(self.output_content_dir, f"{pdf_name}_normalized.txt")
        output_boxes_file = os.path.join(self.output_content_dir, f"{pdf_name}_text_boxes.json")
        
        # 画像変換
        image_paths, _ = self.pdf_to_images(pdf_path)

        # 画像をバッチで処理
        batches = list(self.batch_pdf_to_images(image_paths)[0])
        print(f"バッチ処理を実行します...")
        full_text = ""
        all_text_boxes = []
        
        for i, batch in enumerate(batches):
            print(f"現在の処理中のバッチ：{i+1}")
            
            # バッチ内の各ファイルからページ番号を抽出
            batch_page_info = []
            for path in batch:
                page_num, _ = self.extract_page_number(path)
                if page_num is None:
                    # ファイル名から抽出できない場合はリストの順序を使用
                    page_num = image_paths.index(path) + 1
                batch_page_info.append(f"ページ{page_num}")
            
            batch_page_str = "、".join(batch_page_info)
            print(f"処理中のページ：{batch_page_str}")
            
            # OCR処理（timer_decorator によりタプルで返って来る）
            ocr_result_tuple = self.ocr_complex_document(batch)

            # タプル (result_dict, exec_time) に対応
            ocr_result = ocr_result_tuple[0] if isinstance(ocr_result_tuple, tuple) else ocr_result_tuple
            ocr_time   = ocr_result_tuple[1] if isinstance(ocr_result_tuple, tuple) else None

            # バッチ毎の処理時間も記録（任意）
            if ocr_time is not None:
                self.execution_times[f"{pdf_name}_batch_{i+1}_OCR処理"] = ocr_time

            # 抽出テキスト／テキストボックスを取得
            batch_text = ocr_result["extracted_text"]
            text_boxes = ocr_result["text_boxes"]
            
            all_text_boxes.extend(text_boxes)
            full_text += f"\n\n--- バッチ {i+1} ({batch_page_str}) ---\n\n{batch_text}"
        
        # 全テキストの保存
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(full_text)
            
        # テキストボックスの保存
        with open(output_boxes_file, "w", encoding='utf-8') as f:
            json.dump(all_text_boxes, f, ensure_ascii=False, indent=2)
            
        # テキストの正規化
        normalized_text_result = self.normalize_doc(full_text)
        normalized_text = normalized_text_result[0] if isinstance(normalized_text_result, tuple) else normalized_text_result
        
        # 正規化テキストの保存
        with open(output_file_normalized, "w", encoding='utf-8') as f:
            f.write(normalized_text)
            
        # 各ページのテキストボックスを視覚化
        self.visualize_boxes_for_all_pages(image_paths, all_text_boxes, pdf_name)
        
        return {
            "extracted_text": full_text,
            "normalized_text": normalized_text,
            "text_boxes": all_text_boxes
        }
        
    @timer_decorator
    def visualize_text_boxes(self, image_path, text_boxes, output_path=None, page_num=None):
        """検出されたテキストボックスを視覚化する"""
        # 画像の読み込み
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # フォントを設定（使用可能なフォントがない場合はテキストのみ表示）
        try:
            font = ImageFont.truetype(self.cfg.font.path, self.cfg.font.size)
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
        
    def visualize_boxes_for_all_pages(self, image_paths, all_text_boxes, pdf_name):
        """全ページのテキストボックスを視覚化"""
        # ページごとにボックスを整理
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
                output_viz_path = os.path.join(
                    self.output_visualized_dir, 
                    f"{pdf_name}_visualized_page_{page}.jpg"
                )
                self.visualize_text_boxes(image_path, boxes, output_viz_path, page)
    
    @timer_decorator
    def process_pdf_file(self, pdf_path):
        """単一のPDFファイルを処理"""
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"処理を開始: {pdf_name}")
        
        # 画像変換
        image_paths, img_time = self.pdf_to_images(pdf_path)
        self.execution_times[f"{pdf_name}_画像変換"] = img_time
        
        # ページ数によって処理方法を変更
        start_time = time.time()
        if len(image_paths) > 3:
            print(f"大きなページ数のPDFを検出しました。バッチ処理を実行します...")
            result = self.process_large_pdf(pdf_path)
        else:
            # 少ないページ数の場合は直接処理
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_file = os.path.join(self.output_content_dir, f"{pdf_name}.txt")
            output_file_normalized = os.path.join(self.output_content_dir, f"{pdf_name}_normalized.txt")
            output_boxes_file = os.path.join(self.output_content_dir, f"{pdf_name}_text_boxes.json")
            
            # テキストボックスをページごとに検出
            text_boxes, boxes_time = self.detect_text_boxes(image_paths)
            self.execution_times[f"{pdf_name}_テキストボックス検出"] = boxes_time
            
            # OCRを使用してテキストを抽出
            ocr_result, ocr_time = self.ocr_complex_document(image_paths)
            self.execution_times[f"{pdf_name}_OCR処理"] = ocr_time
            
            extracted_text = ocr_result["extracted_text"]
            
            # テキストを保存
            with open(output_file, "w", encoding='utf-8') as f:
                f.write(extracted_text)
                
            # テキストボックスを保存
            with open(output_boxes_file, "w", encoding='utf-8') as f:
                json.dump(text_boxes, f, ensure_ascii=False, indent=2)
            
            # 各ページを視覚化
            for i, image_path in enumerate(image_paths):
                page_num = i + 1
                output_viz_path = os.path.join(
                    self.output_visualized_dir, 
                    f"{pdf_name}_visualized_page_{page_num}.jpg"
                )
                self.visualize_text_boxes(image_path, text_boxes, output_viz_path, page_num)
            
            # テキストの正規化
            normalized_text, norm_time = self.normalize_doc(extracted_text)
            self.execution_times[f"{pdf_name}_テキスト正規化"] = norm_time
            
            # 保存
            with open(output_file_normalized, "w", encoding='utf-8') as f:
                f.write(normalized_text)
                
            result = {
                "extracted_text": extracted_text,
                "normalized_text": normalized_text,
                "text_boxes": text_boxes
            }
        
        end_time = time.time()
        total_time = end_time - start_time
        self.execution_times[f"{pdf_name}_総処理時間"] = total_time
        
        print(f"😆処理が完了しました🎉\nPDF: {pdf_name}")
        print(f"出力ディレクトリ: {self.output_content_dir}")
        return result
        
    def process_pdf_directory(self):
        """ディレクトリ内の全PDFファイルを処理"""
        pdf_directory = self.cfg.pdf.directory
        pdf_files = glob.glob(os.path.join(pdf_directory, "*.pdf"))
        
        results = {}
        for pdf_file in pdf_files:
            result = self.process_pdf_file(pdf_file)
            results[pdf_file] = result
            
        return results
        
    def print_execution_times(self):
        """処理時間の要約を出力"""
        print("\n=== 処理時間の要約 ===")
        for process, time_taken in self.execution_times.items():
            print(f"{process}: {time_taken:.2f}秒")
            
            # 処理時間の合計をself.output_content_dirに保存
            with open(os.path.join(self.output_content_dir, "execution_times.txt"), "a") as f:
                f.write(f"{process}: {time_taken:.2f}秒\n")
            

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Hydraを使用したメイン関数"""
    print("設定:")
    print(OmegaConf.to_yaml(cfg))
    
    processor = OCRProcessor(cfg)
    
    if cfg.pdf.mode == "single":
        processor.process_pdf_file(cfg.pdf.path)
    elif cfg.pdf.mode == "directory":
        processor.process_pdf_directory()
    else:
        print(f"無効なPDF処理モード: {cfg.pdf.mode}")
        
    processor.print_execution_times()
    
    print("処理が完了しました。")

if __name__ == "__main__":
    main()