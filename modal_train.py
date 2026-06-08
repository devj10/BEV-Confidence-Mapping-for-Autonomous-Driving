import modal

app = modal.App("cs231n-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["libgl1", "libglib2.0-0", "awscli"])
    .pip_install([
        "ultralytics",
        "torch",
        "torchvision",
        "wandb",
        "dropblock",
        "albumentations",
        "nuscenes-devkit",
        "pyquaternion",
        "pyyaml",
        "tqdm",
        "opencv-python-headless", 
    ])
    .add_local_dir(".", remote_path="/root/cs231n")
)

volume = modal.Volume.from_name("cs231n-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60 * 3,
    volumes={"/root/outputs": volume},
)
def train():
    import subprocess, sys, os, shutil, yaml

    os.chdir("/root/cs231n")
    sys.path.insert(0, "/root/cs231n")

    yaml_path = "/root/outputs/yolo_out/dataset.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = "/root/outputs/yolo_out"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f)

    subprocess.run([
        "python", "scripts/train_augmented.py",
        "--data",     yaml_path,
        "--epochs",   "100",
        "--patience", "15",
        "--batch",    "16",
        "--device",   "0",
        "--no-wandb",
    ], check=True)

    shutil.copy(
        "/root/cs231n/results/checkpoints/model_final.pt",
        "/root/outputs/model_final.pt"
    )
    print("Checkpoint saved to volume.")

@app.function(
    image=image,
    timeout=60 * 60 * 8,
    volumes={"/root/outputs": volume},
)
def download_and_convert():
    import subprocess, os

    os.makedirs("/root/outputs/nuscenes", exist_ok=True)

    # Camera-only blobs (~170GB total, vs 300GB for full blobs)
    # Start with 01-03 first (~50GB) — add more if you want
    files = [
        "v1.0-trainval_meta.tgz",            # metadata, small, always needed
        "v1.0-trainval01_blobs_camera.tgz",  # ~17GB
        "v1.0-trainval02_blobs_camera.tgz",  # ~16GB
        "v1.0-trainval03_blobs_camera.tgz",  # ~16GB
    ]

    for f in files:
        print(f"Downloading {f}...")
        subprocess.run([
            "aws", "s3", "cp",
            f"s3://motional-nuscenes/public/v1.0/{f}",
            "/root/outputs/nuscenes/",
            "--no-sign-request"
        ], check=True)

        print(f"Unpacking {f}...")
        subprocess.run([
            "tar", "-xzf", f"/root/outputs/nuscenes/{f}",
            "-C", "/root/outputs/nuscenes/"
        ], check=True)

        os.remove(f"/root/outputs/nuscenes/{f}")  # free up space immediately
        print(f"Done with {f}")

    # Convert to YOLO
    print("Converting to YOLO format...")
    subprocess.run([
        "python", "/root/cs231n/data/nuscenes_to_yolo.py",
        "--dataroot", "/root/outputs/nuscenes",
        "--version",  "v1.0-trainval",
        "--output",   "/root/outputs/yolo_full",
        "--val-fraction", "0.1",
        "--clear-only",
    ], check=True)

    volume.commit()
    print("All done.")

@app.function(
    image=image,
    timeout=60 * 60 * 2,
    volumes={"/root/outputs": volume},
)
def convert_only():
    import subprocess
    subprocess.run([
        "python", "/root/cs231n/data/nuscenes_to_yolo.py",
        "--dataroot", "/root/outputs/nuscenes",
        "--version",  "v1.0-trainval",
        "--output",   "/root/outputs/yolo_full",
        "--val-fraction", "0.1",
        "--clear-only",
        "--no-copy",    # ← add this flag
    ], check=True)
    volume.commit()


@app.local_entrypoint()
def main():
    convert_only.remote()