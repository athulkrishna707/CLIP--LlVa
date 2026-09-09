import json
import tempfile
from pathlib import Path
import pytest

from prepare_dataset import generate_dataset


def test_data_prep_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "data"
        num_samples = 50

        generate_dataset(num_samples=num_samples, output_dir=output_path, seed=42)

        manifest_file = output_path / "manifest.json"
        assert manifest_file.exists(), "manifest.json should be created"

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert len(manifest) == num_samples, f"Expected {num_samples} samples in manifest, got {len(manifest)}"

        # Check split ratios
        train_count = sum(1 for item in manifest if item["split"] == "train")
        val_count = sum(1 for item in manifest if item["split"] == "val")
        test_count = sum(1 for item in manifest if item["split"] == "test")

        assert train_count + val_count + test_count == num_samples
        assert val_count > 0 and test_count > 0

        # Check deduplication & required keys
        image_ids = [item["image_id"] for item in manifest]
        assert len(image_ids) == len(set(image_ids)), "Image IDs must be unique"

        for item in manifest:
            assert "image_id" in item
            assert "image_path" in item
            assert "title" in item
            assert "split" in item
            assert "md5" in item
