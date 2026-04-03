"""S3 upload / download utilities."""

from pathlib import Path
import boto3


def upload_directory_to_s3(
    local_dir: Path, bucket_name: str, s3_prefix: str = ""
) -> tuple:
    """Upload entire directory to S3, preserving structure.

    Args:
        local_dir: Local directory to upload.
        bucket_name: S3 bucket name.
        s3_prefix: Prefix for S3 keys (e.g. ``'datasets/'``).

    Returns:
        ``(uploaded, failed)`` counts.
    """
    s3_client = boto3.client("s3")

    files = [f for f in Path(local_dir).rglob("*") if f.is_file()]
    print(f"Found {len(files)} files to upload...")

    uploaded = 0
    failed = 0

    content_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".json": "application/json",
        ".txt": "text/plain",
        ".mp4": "video/mp4",
    }

    for file_path in files:
        relative_path = file_path.relative_to(local_dir)
        s3_key = f"{s3_prefix}{relative_path}".replace("\\", "/")
        ct = content_types.get(file_path.suffix.lower(), "application/octet-stream")

        try:
            s3_client.upload_file(
                str(file_path), bucket_name, s3_key, ExtraArgs={"ContentType": ct}
            )
            uploaded += 1
            if uploaded % 10 == 0:
                print(f"Uploaded {uploaded}/{len(files)} files...")
        except Exception:
            failed += 1
            print(f"Failed to upload: {file_path}")

    print(f"\n✓ Upload complete: {uploaded} successful, {failed} failed")
    return uploaded, failed


def download_from_s3(
    bucket_name: str, s3_prefix: str, local_dir: Path
) -> int:
    """Download dataset from S3 to local directory.

    Args:
        bucket_name: S3 bucket name.
        s3_prefix: S3 prefix to download from.
        local_dir: Local directory to save files.

    Returns:
        Number of files downloaded.
    """
    s3_client = boto3.client("s3")
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from s3://{bucket_name}/{s3_prefix}...")

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

    files = []
    for page in pages:
        if "Contents" in page:
            files.extend(page["Contents"])

    print(f"Found {len(files)} files to download...")

    downloaded = 0
    for obj in files:
        s3_key = obj["Key"]
        relative_path = s3_key.replace(s3_prefix, "", 1).lstrip("/")
        local_path = local_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        s3_client.download_file(bucket_name, s3_key, str(local_path))
        downloaded += 1

        if downloaded % 10 == 0:
            print(f"Downloaded {downloaded}/{len(files)} files...")

    print(f"\n✓ Download complete: {downloaded} files")
    return downloaded
