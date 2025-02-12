import argparse
import os
import yaml

from ultralytics import YOLO
from google.cloud import storage  # 追加


LOCAL_DATASET_PATH = "./data.yaml"


def download_dataset(bucket_name, dataset_path):
    """Cloud Storage からローカルにデータセットをダウンロード (Vertex AI では不要)"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=dataset_path)  # データセット全体のオブジェクトを取得

    for blob in blobs:
        # ローカルでのパスを構築 (Cloud Storage上のパスから、バケット名とデータセットパスを取り除く)
        local_path = blob.name.replace(dataset_path, "").lstrip("/")
        local_file_path = os.path.join("./", local_path)  # ローカルのカレントディレクトリに保存

        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # ファイルをダウンロード
        blob.download_to_filename(local_file_path)
        print(f"Downloaded {blob.name} to {local_file_path}")


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

    # セグメンテーションモデルを初期化
    if not args.model.endswith("-seg.pt"):
        base_name = args.model.replace(".pt", "")
        args.model = f"{base_name}-seg.pt"

    model = YOLO(args.model)

    # トレーニングを実行
    results = model.train(
        data=LOCAL_DATASET_PATH,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        val=True,  # セグメンテーションの評価を有効化
        iou=args.iou_threshold,  # IoUのしきい値
        conf=args.conf_threshold,  # 信頼度のしきい値
        single_cls=args.single_cls,  # 単一クラスモード
        rect=args.rect,  # 矩形トレーニング
        cos_lr=args.cos_lr,  # コサイン学習率スケジューラー
        mosaic=args.mosaic,  # モザイクデータ拡張
        degrees=args.degrees,  # 回転のデータ拡張
        scale=args.scale,  # スケーリングのデータ拡張
    )

    # 学習済みモデルを保存 (ローカル)
    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"ローカルでのテストのため、モデルは{best_model_path}に保存されました")

    # 学習済みモデルをCloud Storageにアップロード
    if args.upload_bucket:  # --upload_bucket 引数が指定された場合のみアップロード
        upload_model(
            args.upload_bucket,
            best_model_path,
            os.path.join(args.upload_dir, "best.pt"),  # Cloud Storage上のパス
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg on Vertex AI")

    # 基本的なパラメータ
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="Segmentation model to train (e.g., yolov8n-seg.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")

    # セグメンテーション特有のパラメータ
    parser.add_argument(
        "--iou_threshold", type=float, default=0.7, help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--conf_threshold", type=float, default=0.25, help="Confidence threshold"
    )
    parser.add_argument(
        "--single_cls", action="store_true", help="Train as single-class dataset"
    )

    # データ拡張パラメータ
    parser.add_argument("--rect", action="store_true", help="Rectangular training")
    parser.add_argument(
        "--cos_lr", action="store_true", help="Use cosine learning rate scheduler"
    )
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation")
    parser.add_argument(
        "--degrees", type=float, default=0.0, help="Image rotation (+/- deg)"
    )
    parser.add_argument("--scale", type=float, default=0.5, help="Image scale (+/- gain)")

    # Cloud Storage 関連のパラメータ (追加)
    parser.add_argument(
        "--upload_bucket", type=str, help="GCS bucket to upload the trained model to"
    )
    parser.add_argument(
        "--upload_dir",
        type=str,
        default="trained_models",
        help="Directory in the GCS bucket to upload the model to",
    )

    args = parser.parse_args()

    train_yolov8(args)