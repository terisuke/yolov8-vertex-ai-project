import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import argparse
import random

def load_yaml(file_path):
    """YAMLファイルを読み込む"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """バウンディングボックスを描画"""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def plot_segment(img, segments, color=None, label=None, line_thickness=3):
    """セグメンテーションマスクを描画"""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    
    # ポリゴン座標をint32に変換
    points = np.array(segments, dtype=np.int32)
    
    # ポリゴンを描画
    cv2.polylines(img, [points], True, color, thickness=tl, lineType=cv2.LINE_AA)
    
    # ラベルを描画（ポリゴンの左上に配置）
    if label:
        tf = max(tl - 1, 1)
        x_min, y_min = np.min(points[:, 0]), np.min(points[:, 1])
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c1 = (int(x_min), int(y_min))
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def visualize_dataset(data_yaml, num_samples=5, output_dir='visualization_results'):
    """データセットの画像とアノテーションを視覚化"""
    # データ設定を読み込み
    data_config = load_yaml(data_yaml)
    class_names = data_config['names']
    print(f"クラス名: {class_names}")
    
    # トレーニングデータディレクトリを取得
    if isinstance(data_config['train'], list):
        train_dir = data_config['train'][0]
    else:
        train_dir = data_config['train']
    
    print(f"設定されたトレーニングディレクトリ: {train_dir}")
    
    # 画像とラベルのディレクトリ構造を検出
    # YOLOv8の標準的なディレクトリ構造を確認
    possible_img_dirs = [
        os.path.join(train_dir, 'images'),  # 標準的なYOLOv8構造
        train_dir,                           # 直接画像がある場合
    ]
    
    img_dir = None
    for dir_path in possible_img_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            img_dir = dir_path
            break
    
    if img_dir is None:
        print(f"エラー: 画像ディレクトリが見つかりません。次のパスを確認しました: {possible_img_dirs}")
        return
    
    # 対応するラベルディレクトリを探す
    possible_label_dirs = [
        os.path.join(os.path.dirname(img_dir), 'labels'),  # 標準的なYOLOv8構造
        os.path.join(train_dir, 'labels'),                 # 別の一般的な構造
        os.path.join(os.path.dirname(train_dir), 'labels') # さらに別の可能性
    ]
    
    label_dir = None
    for dir_path in possible_label_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            label_dir = dir_path
            break
    
    if label_dir is None:
        print(f"エラー: ラベルディレクトリが見つかりません。次のパスを確認しました: {possible_label_dirs}")
        print("ラベルディレクトリの場所を手動で指定するか、ディレクトリ構造を確認してください。")
        return
    
    print(f"使用する画像ディレクトリ: {img_dir}")
    print(f"使用するラベルディレクトリ: {label_dir}")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 画像リストを取得
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not img_files:
        print(f"エラー: {img_dir} ディレクトリに画像ファイルが見つかりません。")
        return
    
    img_files = img_files[:num_samples]  # サンプル数を制限
    
    # 各画像とそのアノテーションを処理
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(label_path):
            print(f"ラベルファイルが見つかりません: {label_path}")
            continue
        
        # 画像を読み込み
        img = cv2.imread(img_path)
        if img is None:
            print(f"画像を読み込めませんでした: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w, _ = img.shape
        
        # アノテーションを読み込み
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # 各アノテーションを処理
        for line in lines:
            line = line.strip().split()
            if len(line) < 5:  # 最小でもクラスIDとx,y,w,hが必要
                continue
                
            class_id = int(line[0])
            class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown-{class_id}"
            
            # YOLOフォーマットの座標をピクセル座標に変換
            if len(line) == 5:  # バウンディングボックス形式
                x_center, y_center, width, height = map(float, line[1:5])
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                plot_one_box([x1, y1, x2, y2], img, label=f"{class_name} (ID:{class_id})")
            else:  # セグメンテーション形式
                # セグメントポイントをピクセル座標に変換
                segments = []
                for i in range(1, len(line), 2):
                    if i+1 < len(line):
                        x = float(line[i]) * w
                        y = float(line[i+1]) * h
                        segments.append([x, y])
                
                if segments:
                    plot_segment(img, segments, label=f"{class_name} (ID:{class_id})")
        
        # 可視化結果を保存
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.title(f"Image: {img_file}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"vis_{os.path.splitext(img_file)[0]}.png"))
        plt.close()
        
        print(f"画像を処理しました: {img_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8データセットを視覚化")
    parser.add_argument("--data_yaml", type=str, default="data.yaml", help="data.yamlファイルのパス")
    parser.add_argument("--num_samples", type=int, default=5, help="表示するサンプル数")
    parser.add_argument("--output_dir", type=str, default="visualization_results", help="出力ディレクトリ")
    args = parser.parse_args()
    
    visualize_dataset(args.data_yaml, args.num_samples, args.output_dir)
    print(f"可視化結果は {args.output_dir} ディレクトリに保存されました。") 