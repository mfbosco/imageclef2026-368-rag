import os
import json
from pathlib import Path
import json
import re
from pathlib import Path

def extract_number(filename):
    """
    Extract the last number from filename.
    Example:
        ImageCLEFmedical_Caption_2026_valid_12.jpg -> 12
    """
    match = re.search(r'(\d+)(?=\.[^.]+$)', filename)
    return int(match.group(1)) if match else -1


def create_test_json(
    image_dir,
    output_json,
    extensions={".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
):
    image_dir = Path(image_dir)

    # Filter valid images first
    image_paths = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    ]

    # Sort by extracted number
    image_paths.sort(key=lambda p: extract_number(p.name))

    data = []

    for img_path in image_paths:
        entry = {
            "image_path": str(img_path.resolve()),
            "image_id": img_path.stem
        }
        data.append(entry)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} entries to {output_json}")


if __name__ == "__main__":
    create_test_json(
        image_dir="/home/ia368/projetos/imageclef2026-rag/dev_caption/test/images",  # <-- change this
        output_json="/home/ia368/projetos/imageclef2026-rag/artifacts/datasets/imageclef2026_test_dataset.json"
    )