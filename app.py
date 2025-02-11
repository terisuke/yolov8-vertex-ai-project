import argparse
import os
import yaml

from ultralytics import YOLO
#from google.cloud import storage  # ローカルテスト時はコメントアウト

# Cloud Storage バケット名 (バケット名は適宜変更してください)
# BUCKET_NAME = "yolo-v8-training"  # ローカルテスト時はコメントアウト

# データセットのパス (Cloud Storage上のパス)
# DATASET_PATH = "datasets/your-dataset.yaml"  # ローカルテスト時はコメントアウト
# ローカルデータセットのパス（ローカルテスト時用）
LOCAL_DATASET_PATH = "./data.yaml"

def download_dataset(bucket_name, dataset_path):
    """Cloud Storageからデータセットをダウンロードする関数"""
    # client = storage.Client()
    # bucket = client.bucket(bucket_name)

    # # データセットのyamlファイル (例: data.yaml) をダウンロード
    # blob = bucket.blob(dataset_path)
    # blob.download_to_filename(os.path.basename(dataset_path))

    # # データセットの画像とラベルをダウンロード (ディレクトリ構造を維持)
    # # data.yamlからtrainとvalのパスを取得
    # with open(os.path.basename(dataset_path), 'r') as f:
    #     data_yaml = yaml.safe_load(f)
    #     train_path = data_yaml['train']
    #     val_path = data_yaml['val']
    
    # # train data
    # blobs = bucket.list_blobs(prefix= train_path)  # Get list of all blobs
    # for blob in blobs:
    #     if blob.name.endswith((".jpg", ".jpeg", ".png")):
    #         destination_file_name = blob.name.replace(train_path,"data/images/train")
    #         os.makedirs(os.path.dirname(destination_file_name),exist_ok=True) # ディレクトリがない場合は作成
    #         blob.download_to_filename(destination_file_name)
            
    # blobs = bucket.list_blobs(prefix= train_path.replace("images","labels"))  # Get list of all blobs
    # for blob in blobs:
    #     if blob.name.endswith((".txt")):
    #         destination_file_name = blob.name.replace(train_path,"data/labels/train")
    #         os.makedirs(os.path.dirname(destination_file_name),exist_ok=True) # ディレクトリがない場合は作成
    #         blob.download_to_filename(destination_file_name)
    # # val data
    # blobs = bucket.list_blobs(prefix= val_path)  # Get list of all blobs
    # for blob in blobs:
    #     if blob.name.endswith((".jpg", ".jpeg", ".png")):
    #         destination_file_name = blob.name.replace(val_path,"data/images/val")
    #         os.makedirs(os.path.dirname(destination_file_name),exist_ok=True) # ディレクトリがない場合は作成
    #         blob.download_to_filename(destination_file_name)
            
    # blobs = bucket.list_blobs(prefix= val_path.replace("images","labels"))  # Get list of all blobs
    # for blob in blobs:
    #     if blob.name.endswith((".txt")):
    #         destination_file_name = blob.name.replace(val_path,"data/labels/val")
    #         os.makedirs(os.path.dirname(destination_file_name),exist_ok=True) # ディレクトリがない場合は作成
    #         blob.download_to_filename(destination_file_name)
    pass


def upload_model(bucket_name, model_path, destination_blob_name):
    """学習済みモデルをCloud Storageにアップロードする関数"""
    # client = storage.Client()
    # bucket = client.bucket(bucket_name)
    # blob = bucket.blob(destination_blob_name)
    # blob.upload_from_filename(model_path)
    pass # ローカルでのテスト時は何もしない

def train_yolov8(args):
    """YOLOv8のトレーニングを実行する関数"""

    # データセットをダウンロード
    # download_dataset(BUCKET_NAME, DATASET_PATH) # ローカルテスト時はコメントアウト

    # YOLOモデルを初期化
    model = YOLO(args.model)  # 例: YOLO("yolov8n.pt")

    # トレーニングを実行
    results = model.train(
        # data=os.path.basename(DATASET_PATH),  # ローカルテスト時はコメントアウト
        data=LOCAL_DATASET_PATH, # ローカルのデータセットパス
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        optimizer = args.optimizer,
        val=False,  # 検証データの読み込みを無効化
         # その他のハイパーパラメータ...
    )

    # 学習済みモデルをCloud Storageにアップロード
    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    # upload_model(BUCKET_NAME, best_model_path, f"models/{args.model}/best.pt") # ローカルテスト時はコメントアウト
    print(f"ローカルでのテストのため、モデルは{best_model_path}に保存されました")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Vertex AI")

    # コマンドライン引数を定義
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model to train (e.g., yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--optimizer",type=str, default="Adam",help="optimizer")
    # その他の引数を追加...

    args = parser.parse_args()

    train_yolov8(args)