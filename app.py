import argparse
import os
import yaml
from google.cloud import storage
from ultralytics import YOLO

def download_dataset(bucket_name, source_prefix, destination_dir):
    """Downloads dataset from Cloud Storage, maintaining directory structure and skipping unwanted files."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Ensure the prefix ends with '/' for consistency
    if not source_prefix.endswith('/'):
        source_prefix += '/'

    # List blobs, excluding .DS_Store and labels.cache
    blobs = [blob for blob in bucket.list_blobs(prefix=source_prefix)
             if not blob.name.endswith(('.DS_Store', 'labels.cache'))]

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    for blob in blobs:
         # Extract relative path and construct the destination file path
        relative_path = blob.name.replace(source_prefix, '', 1)
        destination_file = os.path.join(destination_dir, relative_path)

        # Create any necessary subdirectories within the destination
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)

        # Download only files (not directories, which are represented as prefixes)
        if blob.name.endswith('/'):  # Skip directories
            continue
            
        print(f"Downloading: {blob.name} -> {destination_file}") # show progress
        blob.download_to_filename(destination_file)

def upload_model(bucket_name, model_path, destination_blob_name):
    """Uploads the trained model to Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(model_path)
    print(f"File {model_path} uploaded to gs://{bucket_name}/{destination_blob_name}")


def train_yolov8(args):
    """Trains a YOLOv8 segmentation model."""

    # Download dataset first (if on Vertex AI) and update data.yaml
    if args.bucket_name:  # If bucket_name is provided, assume we're on Vertex AI
        base_dir = "/app/datasets/house"
        train_dir = os.path.join(base_dir, "train")
        val_dir = os.path.join(base_dir, "val")

        # Create the necessary directory structure
        for dir_path in [
            os.path.join(train_dir, "images"),
            os.path.join(train_dir, "labels"),
            os.path.join(val_dir, "images"),
            os.path.join(val_dir, "labels")
        ]:
            os.makedirs(dir_path, exist_ok=True)

        # Download training data, now correctly separating images and labels
        download_dataset(args.bucket_name, "house/train/images", os.path.join(train_dir, "images"))
        download_dataset(args.bucket_name, "house/train/labels", os.path.join(train_dir, "labels"))

        # Download validation data, separating images and labels
        download_dataset(args.bucket_name, "house/val/images", os.path.join(val_dir, "images"))
        download_dataset(args.bucket_name, "house/val/labels", os.path.join(val_dir, "labels"))

         # Check downloaded contents (for debugging - can be removed in production)
        print("Training directory structure:")
        print(f"Images: {os.listdir(os.path.join(train_dir, 'images'))}")
        print(f"Labels: {os.listdir(os.path.join(train_dir, 'labels'))}")

        print("\nValidation directory structure:")
        print(f"Images: {os.listdir(os.path.join(val_dir, 'images'))}")
        print(f"Labels: {os.listdir(os.path.join(val_dir, 'labels'))}")


        data_path = "/app/data.yaml"

        # Update data.yaml with local paths
        with open(data_path, "r") as f:
            data_config = yaml.safe_load(f)

        # Set paths as expected by YOLOv8
        data_config["train"] = train_dir
        data_config["val"] = val_dir
        # data_config['path'] = '/app/datasets/house'  # Optional: Set a base path if needed


        with open(data_path, "w") as f:
            yaml.dump(data_config, f)

    else:  # Local execution
        data_path = args.data_yaml_local_path
        with open(data_path, 'r') as f:
            data_config = yaml.safe_load(f)

            data_config['train'] = args.train_dir
            data_config['val'] = args.val_dir

            with open(data_path, 'w') as f:
                yaml.dump(data_config, f)


    # Initialize the segmentation model
    if not args.model.endswith("-seg.pt"):
        base_name = args.model.replace(".pt", "")
        args.model = f"{base_name}-seg.pt"

    model = YOLO(args.model)

    # Perform training
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        val=True, # Enable validation
        iou=args.iou_threshold,
        conf=args.conf_threshold,
        single_cls=args.single_cls,
        rect=args.rect,
        cos_lr=args.cos_lr,
        mosaic=args.mosaic,
        degrees=args.degrees,
        scale=args.scale,
        exist_ok=True  # Allow overwriting previous runs
    )


    # Save the trained model (local)
    best_model_path = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    print(f"Model saved to {best_model_path}")

    # Upload the trained model to Cloud Storage
    if args.upload_bucket:
        upload_model(
            args.upload_bucket,
            best_model_path,
            os.path.join(args.upload_dir, "best.pt"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg on Vertex AI")

    # Basic parameters
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Base YOLOv8 model (e.g., yolov8n-seg.pt, yolov8s-seg.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer (e.g., Adam, SGD)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")

    # Segmentation-specific parameters
    parser.add_argument("--iou_threshold", type=float, default=0.7, help="IoU threshold for segmentation")
    parser.add_argument("--conf_threshold", type=float, default=0.25, help="Confidence threshold for object detection")
    parser.add_argument("--single_cls", action="store_true", help="Treat all classes as a single class")

    # Data augmentation parameters
    parser.add_argument("--rect", action="store_true", help="Enable rectangular training")
    parser.add_argument("--cos_lr", action="store_true", help="Use cosine learning rate scheduler")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")
    parser.add_argument("--degrees", type=float, default=0.0, help="Image rotation degrees")
    parser.add_argument("--scale", type=float, default=0.5, help="Image scale augmentation")

    # Cloud Storage related parameters
    parser.add_argument("--upload_bucket", type=str, help="GCS bucket to upload the trained model to")
    parser.add_argument("--upload_dir", type=str, default="trained_models", help="Directory in the bucket to upload to")

    # data.yaml GCS path (required in Vertex AI environment)
    parser.add_argument("--bucket_name", type=str, help="GCS bucket name for dataset (required on Vertex AI)")  # Required for Vertex AI
    parser.add_argument("--data_yaml_gcs_path", type=str, help="GCS path to data.yaml (required on Vertex AI)") # Vertex AIなら必須

     # data.yaml and train/val local paths (used for local execution)
    parser.add_argument('--data_yaml_local_path', type=str, default= "data.yaml", help="Local path to data.yaml")
    parser.add_argument('--train_dir', type=str, default= "datasets/house/train", help='Local path to the training data directory') # ローカル用
    parser.add_argument('--val_dir', type=str, default= "datasets/house/val", help='Local path to the validation data directory') # ローカル用


    args = parser.parse_args()

    train_yolov8(args)