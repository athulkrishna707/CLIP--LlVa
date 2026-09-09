import json
import tempfile
from pathlib import Path
import pytest

from caption_generation import run_caption_generation


def test_caption_cache_and_resumability():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create dummy manifest
        manifest_path = data_dir / "manifest.json"
        manifest = [
            {"image_id": "prod_00001", "image_path": str(data_dir / "img1.jpg"), "title": "Test Shoe", "category": "footwear", "split": "train"},
            {"image_id": "prod_00002", "image_path": str(data_dir / "img2.jpg"), "title": "Test Watch", "category": "electronics", "split": "train"},
        ]
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        cache_path = data_dir / "captions.json"

        # 1. Run captioning initial pass
        run_caption_generation(manifest_path, cache_path, model_name="none", device="cpu")

        assert cache_path.exists(), "Caption cache JSON should be created"

        with open(cache_path, "r", encoding="utf-8") as f:
            captions = json.load(f)

        assert "prod_00001" in captions
        assert "prod_00002" in captions
        assert "Test Shoe" in captions["prod_00001"]["caption"] or "footwear" in captions["prod_00001"]["caption"].lower()

        # Modify one caption to test resumable skip
        captions["prod_00001"]["caption"] = "PRESERVED_CUSTOM_CAPTION"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(captions, f)

        # 2. Run captioning second pass (resumability check)
        run_caption_generation(manifest_path, cache_path, model_name="none", device="cpu")

        with open(cache_path, "r", encoding="utf-8") as f:
            updated_captions = json.load(f)

        assert updated_captions["prod_00001"]["caption"] == "PRESERVED_CUSTOM_CAPTION", "Resumable runner should not overwrite existing cached captions"
