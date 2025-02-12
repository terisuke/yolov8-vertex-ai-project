import argparse
import os
import yaml

from ultralytics import YOLO

LOCAL_DATASET_PATH = "./data.yaml"

def download_dataset(bucket_name, dataset_path):
    pass

def upload_model(bucket_name, model_path, destination_blob_name):
    pass

def train_yolov8(args):
    """YOLOv8のセグメンテーションモデルのトレーニングを実行する関数"""
    
    # セグメンテーションモデルを初期化
    if not args.model.endswith('-seg.pt'):
        base_name = args.model.replace('.pt', '')
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

    # 学習済みモデルを保存
    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"ローカルでのテストのため、モデルは{best_model_path}に保存されました")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg on Vertex AI")

    # 基本的なパラメータ
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Segmentation model to train (e.g., yolov8n-seg.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    
    # セグメンテーション特有のパラメータ
    parser.add_argument("--iou_threshold", type=float, default=0.7, help="IoU threshold for NMS")
    parser.add_argument("--conf_threshold", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--single_cls", action="store_true", help="Train as single-class dataset")
    
    # データ拡張パラメータ
    parser.add_argument("--rect", action="store_true", help="Rectangular training")
    parser.add_argument("--cos_lr", action="store_true", help="Use cosine learning rate scheduler")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation")
    parser.add_argument("--degrees", type=float, default=0.0, help="Image rotation (+/- deg)")
    parser.add_argument("--scale", type=float, default=0.5, help="Image scale (+/- gain)")

    args = parser.parse_args()

    train_yolov8(args)