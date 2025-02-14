import argparse
import os
import yaml

from ultralytics import YOLO
from google.cloud import storage

def download_dataset(bucket_name, source_prefix, destination_dir):
    """Downloads dataset from Cloud Storage, maintaining directory structure."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Ensure the prefix ends with '/'
    if not source_prefix.endswith('/'):
        source_prefix += '/'

    blobs = bucket.list_blobs(prefix=source_prefix, delimiter='/')

    os.makedirs(destination_dir, exist_ok=True)

    for blob in blobs:
        if isinstance(blob, storage.blob.Blob):
            relative_path = blob.name.replace(source_prefix, '', 1)
            destination_file = os.path.join(destination_dir, relative_path)
            os.makedirs(os.path.dirname(destination_file), exist_ok=True)
            blob.download_to_filename(destination_file)
        elif blob.prefixes:
            for sub_prefix in blob.prefixes:
                sub_destination_dir = os.path.join(destination_dir, sub_prefix.replace(source_prefix, '', 1))
                download_dataset(bucket_name, sub_prefix, sub_destination_dir)


def upload_model(bucket_name, model_path, destination_blob_name):
    """学習済みモデルをCloud Storageにアップロード"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(model_path)
    print(
        f"File {model_path} uploaded to gs://{bucket_name}/{destination_blob_name}"
    )


def train_yolov8(args):
    """YOLOv8のセグメンテーションモデルのトレーニングを実行する関数"""

    # Download dataset first (if on Vertex AI) and update data.yaml
    if args.bucket_name:  # bucket_name が指定されていれば、Vertex AI 上と判断
        train_dir = "/app/datasets/house/train"
        val_dir = "/app/datasets/house/val"

        download_dataset(args.bucket_name, "house/train", train_dir)
        download_dataset(args.bucket_name, "house/val", val_dir)

        data_path = "/app/data.yaml"

        # Update data.yaml with local paths
        with open("data.yaml", "r") as f:
            data_config = yaml.safe_load(f)

        data_config["train"] = train_dir
        data_config["val"] = val_dir

        with open(data_path, "w") as f:
            yaml.dump(data_config, f)

    else:  # ローカル実行
        data_path = args.data_yaml_local_path
        with open("data.yaml", "r") as f:
                data_config = yaml.safe_load(f)

        data_config["train"] = args.train_dir
        data_config["val"] = args.val_dir

        with open(data_path, "w") as f:
            yaml.dump(data_config, f)



    # セグメンテーションモデルを初期化
    if not args.model.endswith("-seg.pt"):
        base_name = args.model.replace(".pt", "")
        args.model = f"{base_name}-seg.pt"

    model = YOLO(args.model)

    # トレーニングを実行
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        val=True,
        iou=args.iou_threshold,
        conf=args.conf_threshold,
        single_cls=args.single_cls,
        rect=args.rect,
        cos_lr=args.cos_lr,
        mosaic=args.mosaic,
        degrees=args.degrees,
        scale=args.scale,
        exist_ok=True
    )


    # 学習済みモデルを保存 (ローカル)
    best_model_path = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    print(f"モデルは{best_model_path}に保存されました")

    # 学習済みモデルをCloud Storageにアップロード
    if args.upload_bucket:
        upload_model(
            args.upload_bucket,
            best_model_path,
            os.path.join(args.upload_dir, "best.pt"),
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg on Vertex AI")

    # 基本的なパラメータ
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr0", type=float, default=0.01)

    # セグメンテーション特有のパラメータ
    parser.add_argument("--iou_threshold", type=float, default=0.7)
    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument("--single_cls", action="store_true")

    # データ拡張パラメータ
    parser.add_argument("--rect", action="store_true")
    parser.add_argument("--cos_lr", action="store_true")
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.5)

    # Cloud Storage 関連のパラメータ
    parser.add_argument("--upload_bucket", type=str)
    parser.add_argument("--upload_dir", type=str, default="trained_models")

    # data.yaml の GCS パス (Vertex AI 環境で必要)
    parser.add_argument("--bucket_name", type=str) # Vertex AI なら必須
    parser.add_argument("--data_yaml_gcs_path", type=str) # Vertex AI なら必須

     # data.yaml と train/val のローカルパス (ローカル環境での実行時に使用)
    parser.add_argument('--data_yaml_local_path', type=str, default= "data.yaml")
    parser.add_argument('--train_dir', type=str, default= "datasets/house/train") # ローカル用
    parser.add_argument('--val_dir', type=str, default= "datasets/house/val") # ローカル用


    args = parser.parse_args()

    train_yolov8(args)