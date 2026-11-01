import json
import tempfile
from pathlib import Path
import pytest

from prepare_dataset import generate_dataset, prepare_dataset_from_db
from database import init_products_db, insert_or_update_product


def test_prepare_dataset_from_db():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        db_file = tmp_root / "test_products.db"
        output_path = tmp_root / "data"

        init_products_db(db_file)
        # Insert test products
        for i in range(5):
            insert_or_update_product({
                "id": f"TEST_{i+1:03d}",
                "title": f"Test Product {i+1}",
                "description": f"Test description for product {i+1}",
                "category": "electronics",
                "split": "train" if i < 3 else "test"
            }, db_path=db_file)

        manifest_file = prepare_dataset_from_db(output_dir=output_path, db_path=db_file, seed=42)
        assert manifest_file.exists()

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert len(manifest) == 5
        assert all("image_id" in x and "image_path" in x and "md5" in x for x in manifest)



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
