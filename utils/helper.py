from google.cloud import storage
import os

def download_from_gcs(bucket_name, source_prefix, destination_dir):
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

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the specified Cloud Storage bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}")