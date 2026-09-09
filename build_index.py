import argparse
import json
import time
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import faiss
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel


class IndexingProductDataset(Dataset):
    def __init__(self, manifest_items, captions_dict, processor, image_root: Path):
        self.items = manifest_items
        self.captions_dict = captions_dict
        self.processor = processor
        self.image_root = image_root

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_id = item["image_id"]
        img_path = Path(item["image_path"])

        if not img_path.is_absolute():
            img_path = self.image_root / img_path

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(200, 200, 200))

        title = item.get("title", "")
        metadata_text = item.get("metadata_text", title)
        caption = self.captions_dict.get(img_id, {}).get("caption", metadata_text)
        description = item.get("description", self.captions_dict.get(img_id, {}).get("caption", metadata_text))
        category = item.get("category", "")

        return {
            "image": image,
            "image_id": img_id,
            "image_path": str(item["image_path"]),
            "title": title,
            "caption": caption,
            "description": description,
            "category": category
        }


def extract_tensor(feat):
    """Unwraps tensor from CLIP output objects if necessary."""
    if isinstance(feat, torch.Tensor):
        return feat
    if hasattr(feat, "image_embeds") and feat.image_embeds is not None:
        return feat.image_embeds
    if hasattr(feat, "text_embeds") and feat.text_embeds is not None:
        return feat.text_embeds
    if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
        return feat.pooler_output
    if hasattr(feat, "last_hidden_state") and feat.last_hidden_state is not None:
        return feat.last_hidden_state[:, 0, :]
    if isinstance(feat, (list, tuple)):
        return feat[0]
    return feat


def build_faiss_index(
    manifest_path: Path,
    captions_path: Path,
    checkpoint_dir: Path,
    output_index_path: Path,
    output_meta_path: Path,
    base_model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Building FAISS Index for retrieval service on {device}...")
    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(captions_path, "r", encoding="utf-8") as f:
        captions_dict = json.load(f)

    print(f"Total product images to encode & index: {len(manifest)}")

    processor = CLIPProcessor.from_pretrained(base_model_name)
    base_model = CLIPModel.from_pretrained(base_model_name).to(device)

    best_model_path = checkpoint_dir / "best_model"
    if best_model_path.exists():
        try:
            model = PeftModel.from_pretrained(base_model, str(best_model_path)).to(device)
            print(f"Loaded fine-tuned model from {best_model_path} for indexing.")
        except Exception as e:
            print(f"Notice: Loading PeftModel failed ({e}). Falling back to base CLIP model.")
            model = base_model
    else:
        print(f"Checkpoint not found at {best_model_path}. Using base CLIP model for indexing.")
        model = base_model

    model.eval()
    image_root = manifest_path.parent.parent

    dataset = IndexingProductDataset(manifest, captions_dict, processor, image_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: {
        "image": [x["image"] for x in b],
        "image_id": [x["image_id"] for x in b],
        "image_path": [x["image_path"] for x in b],
        "title": [x["title"] for x in b],
        "caption": [x["caption"] for x in b],
        "description": [x["description"] for x in b],
        "category": [x["category"] for x in b]
    })

    all_embeddings = []
    metadata_records = []

    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding Product Images"):
            images = batch["image"]
            inputs = processor(images=images, return_tensors="pt").to(device)
            image_features = extract_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
            image_features = F.normalize(image_features, dim=-1)

            all_embeddings.append(image_features.cpu().numpy())

            for i in range(len(images)):
                metadata_records.append({
                    "vector_id": len(metadata_records),
                    "image_id": batch["image_id"][i],
                    "image_path": batch["image_path"][i],
                    "title": batch["title"][i],
                    "description": batch["description"][i],
                    "llava_caption": batch["caption"][i],
                    "category": batch["category"][i]
                })

    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    embedding_dim = embeddings_matrix.shape[1]
    print(f"Encoded {embeddings_matrix.shape[0]} images. Vector dimension: {embedding_dim}")

    # Build FAISS Index (Inner Product / Cosine Similarity on normalized vectors)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings_matrix)

    faiss.write_index(index, str(output_index_path))
    print(f"FAISS Index written to {output_index_path} ({index.ntotal} vectors).")

    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2)
    print(f"Index Metadata written to {output_meta_path}.")

    elapsed = time.time() - start_time
    print(f"Indexing completed successfully in {elapsed:.2f}s!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAISS Index Building Stage")
    parser.add_argument("--manifest", type=str, default="data/manifest.json", help="Path to manifest JSON")
    parser.add_argument("--captions", type=str, default="data/captions.json", help="Path to captions JSON")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to model checkpoints")
    parser.add_argument("--output_index", type=str, default="index/product_faiss.index", help="Output FAISS index path")
    parser.add_argument("--output_meta", type=str, default="index/index_metadata.json", help="Output index metadata path")
    parser.add_argument("--base_model", type=str, default="openai/clip-vit-base-patch32", help="Base model name")
    args = parser.parse_args()

    build_faiss_index(
        manifest_path=Path(args.manifest),
        captions_path=Path(args.captions),
        checkpoint_dir=Path(args.checkpoint_dir),
        output_index_path=Path(args.output_index),
        output_meta_path=Path(args.output_meta),
        base_model_name=args.base_model
    )
