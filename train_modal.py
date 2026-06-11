"""
train_modal.py

Modal app that downloads nuScenes from Azure, converts it to YOLO format into a
persistent volume, and launches augmented training on an A10G GPU.
"""

import modal

app = modal.App("cs231n-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "libgl1",
        "libglib2.0-0",
        "curl",
        "ca-certificates",
        "tar",
    ])
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


def install_azcopy():
    """Install azcopy into /usr/local/bin if not already present."""
    import subprocess

    subprocess.run(
        [
            "bash",
            "-lc",
            """
            set -e
            if command -v azcopy >/dev/null 2>&1; then
                azcopy --version
                exit 0
            fi

            cd /tmp
            curl -L -o azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
            tar -xzf azcopy.tar.gz
            cp ./azcopy_linux_amd64_*/azcopy /usr/local/bin/azcopy
            chmod +x /usr/local/bin/azcopy
            azcopy --version
            """,
        ],
        check=True,
    )


def azure_prefix_url(container_sas_url: str, prefix: str) -> str:
    """Append a blob prefix to a container-level SAS URL."""
    base, query = container_sas_url.split("?", 1)
    return f"{base}/{prefix.strip('/')}?{query}"


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 12,
    ephemeral_disk=512 * 1024,
    volumes={"/root/outputs": volume},
    secrets=[modal.Secret.from_name("azure-sas")],
)
def train_from_azure_tmp():
    """Download nuScenes from Azure, convert to YOLO, and train; skip download if dataset already exists."""
    import os
    import sys
    import yaml
    import shutil
    import subprocess
    from pathlib import Path

    os.chdir("/root/cs231n")
    sys.path.insert(0, "/root/cs231n")

    def find_nuscenes_root(base: Path, version: str = "v1.0-trainval") -> Path:
        """Search for the nuScenes dataroot containing the given version directory."""
        candidates = [
            base,
            base / "nuscenes",
            base / "data",
            base / "v1.0",
        ]

        for candidate in candidates:
            if (candidate / version).exists():
                return candidate

        matches = list(base.rglob(version))
        if matches:
            return matches[0].parent

        print(f"Could not find {version} under {base}")
        print("Downloaded directory structure:")
        subprocess.run(
            ["bash", "-lc", f"find {base} -maxdepth 4 -type d | head -200"],
            check=False,
        )

        raise FileNotFoundError(f"Could not find {version} under {base}")

    def count_files(path: Path, pattern: str = "*") -> int:
        """Return the number of files matching pattern under path."""
        if not path.exists():
            return 0
        return sum(1 for _ in path.rglob(pattern))

    def count_images(path: Path) -> int:
        """Return the number of image files under path."""
        if not path.exists():
            return 0

        exts = {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".tif", ".tiff", ".heic", ".heif", ".dng",
            ".mpo", ".avif", ".jp2", ".jpeg2000",
        }

        total = 0
        for p in path.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                total += 1
        return total

    def print_yolo_counts(yolo_root: Path):
        """Print train/val image and label counts for the YOLO dataset."""
        images_train = yolo_root / "images" / "train"
        images_val = yolo_root / "images" / "val"
        labels_train = yolo_root / "labels" / "train"
        labels_val = yolo_root / "labels" / "val"

        print("YOLO dataset counts:")
        print(f"  train images: {count_images(images_train)}")
        print(f"  val images:   {count_images(images_val)}")
        print(f"  train labels: {count_files(labels_train, '*.txt')}")
        print(f"  val labels:   {count_files(labels_val, '*.txt')}")

    def ensure_nonempty_val_split(yolo_root: Path, max_val_images: int = 500):
        """Move a fraction of train images to val if the val split is empty."""
        images_train = yolo_root / "images" / "train"
        images_val = yolo_root / "images" / "val"
        labels_train = yolo_root / "labels" / "train"
        labels_val = yolo_root / "labels" / "val"

        images_train.mkdir(parents=True, exist_ok=True)
        images_val.mkdir(parents=True, exist_ok=True)
        labels_train.mkdir(parents=True, exist_ok=True)
        labels_val.mkdir(parents=True, exist_ok=True)

        val_count = count_images(images_val)
        train_count = count_images(images_train)

        if val_count > 0:
            print(f"Validation split already has {val_count} images.")
            return

        if train_count == 0:
            raise RuntimeError(
                "Both train and val image splits are empty. "
                "Conversion produced no usable images."
            )

        print(
            f"Validation split is empty. Moving up to {max_val_images} "
            f"images from train to val."
        )

        image_exts = {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".tif", ".tiff", ".heic", ".heif", ".dng",
            ".mpo", ".avif", ".jp2", ".jpeg2000",
        }

        train_images = [
            p for p in images_train.rglob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        ]

        train_images = sorted(train_images)
        num_to_move = min(max_val_images, max(1, len(train_images) // 10))

        for img_path in train_images[:num_to_move]:
            rel = img_path.relative_to(images_train)

            dst_img = images_val / rel
            dst_img.parent.mkdir(parents=True, exist_ok=True)

            label_src = labels_train / rel.with_suffix(".txt")
            label_dst = labels_val / rel.with_suffix(".txt")
            label_dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(str(img_path), str(dst_img))

            if label_src.exists():
                shutil.move(str(label_src), str(label_dst))
            else:
                label_dst.touch()

        print("Created fallback validation split.")
        print_yolo_counts(yolo_root)

    def fix_dataset_yaml(yolo_root: Path) -> Path:
        """Update the path field in dataset.yaml to match the current yolo_root."""
        yaml_path = yolo_root / "dataset.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"Missing dataset.yaml at {yaml_path}")

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f) or {}

        cfg["path"] = str(yolo_root)

        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f)

        print("Final dataset.yaml:")
        subprocess.run(["cat", str(yaml_path)], check=True)

        return yaml_path

    def yolo_dataset_ready(yolo_root: Path) -> bool:
        """Return True if dataset.yaml exists and both train and val splits are non-empty."""
        yaml_path = yolo_root / "dataset.yaml"

        if not yaml_path.exists():
            return False

        train_count = count_images(yolo_root / "images" / "train")
        val_count = count_images(yolo_root / "images" / "val")

        return train_count > 0 and val_count > 0

    download_root = Path("/tmp/nuscenes")
    yolo_root = Path("/root/outputs/yolo_full")

    yolo_root.mkdir(parents=True, exist_ok=True)

    if yolo_dataset_ready(yolo_root):
        print("Found existing YOLO dataset in Modal volume.")
        print("Skipping Azure raw nuScenes download and conversion.")
        yaml_path = fix_dataset_yaml(yolo_root)
        print_yolo_counts(yolo_root)

    else:
        print("No complete YOLO dataset found in Modal volume.")
        print("Downloading raw nuScenes from Azure and converting once...")

        install_azcopy()

        sas_url = os.environ["AZURE_CONTAINER_SAS_URL"]

        download_root.mkdir(parents=True, exist_ok=True)

        source = azure_prefix_url(sas_url, "nuscenes")

        print("Downloading nuScenes from Azure to /tmp, NOT Modal volume...")
        subprocess.run(
            [
                "azcopy",
                "copy",
                source,
                str(download_root),
                "--recursive=true",
                "--overwrite=ifSourceNewer",
            ],
            check=True,
        )

        print("Disk usage after download:")
        subprocess.run(
            ["bash", "-lc", "df -h /tmp /root/outputs && du -sh /tmp/nuscenes"],
            check=True,
        )

        print("Checking downloaded directory structure...")
        subprocess.run(
            ["bash", "-lc", "find /tmp/nuscenes -maxdepth 3 -type d | head -100"],
            check=True,
        )

        nusc_root = find_nuscenes_root(download_root, "v1.0-trainval")
        print(f"Using nuScenes dataroot: {nusc_root}")

        print("Clearing any incomplete old YOLO dataset...")
        if yolo_root.exists():
            shutil.rmtree(yolo_root)
        yolo_root.mkdir(parents=True, exist_ok=True)

        print("Converting nuScenes to YOLO into Modal volume...")
        subprocess.run(
            [
                "python",
                "/root/cs231n/data/nuscenes_to_yolo.py",
                "--dataroot",
                str(nusc_root),
                "--version",
                "v1.0-trainval",
                "--output",
                str(yolo_root),
                "--val-fraction",
                "0.1",
                "--clear-only",
            ],
            check=True,
        )

        yaml_path = fix_dataset_yaml(yolo_root)

        print("Checking YOLO counts before fallback val split...")
        print_yolo_counts(yolo_root)

        ensure_nonempty_val_split(yolo_root, max_val_images=500)

        print("Checking YOLO counts after fallback val split...")
        print_yolo_counts(yolo_root)

        print("Committing YOLO dataset to Modal volume...")
        volume.commit()
        print("YOLO dataset saved in Modal volume at /root/outputs/yolo_full.")

    print("Disk usage before training:")
    subprocess.run(
        [
            "bash",
            "-lc",
            "df -h /tmp /root/outputs && du -sh /root/outputs/yolo_full",
        ],
        check=True,
    )

    # Oversampling skipped — touching 60K files on a network volume is too slow
    # regardless of approach (copy, grep, iterdir). Class imbalance is handled by:
    #   - cls=1.5 loss upweighting in train_augmented.py
    #   - mosaic augmentation (mixes 4 images, increases rare class encounters)
    print("Starting training...")
    subprocess.run(
    [
        "python",
        "scripts/train_augmented.py",
        "--data",
        str(yaml_path),
        "--epochs",
        "50",
        "--patience",
        "20",
        "--batch",
        "16",
        "--device",
        "0",
        "--project",
        "/root/outputs/results",
        "--name",
        "checkpoints",
        "--export-path",
        "/root/outputs/model_final.pt",
        "--aug-config",
        "configs/augmentation.yaml",
        "--no-wandb",
    ],
    check=True,
    )

    volume.commit()

    print("Checkpoint saved to Modal volume.")
    print("Final disk usage:")
    subprocess.run(
        ["bash", "-lc", "df -h /tmp /root/outputs && ls -lah /root/outputs"],
        check=True,
    )


@app.local_entrypoint()
def main():
    train_from_azure_tmp.remote()
