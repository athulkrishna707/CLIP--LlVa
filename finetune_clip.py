import argparse
import json
import math
import os
import time
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model


class ProductDataset(Dataset):
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
        llava_caption = self.captions_dict.get(img_id, {}).get("caption", metadata_text)

        return {
            "image": image,
            "llava_caption": llava_caption,
            "metadata_text": metadata_text,
            "image_id": img_id
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


def info_nce_loss(image_embeds, text_embeds, logit_scale):
    """Computes InfoNCE symmetric contrastive loss."""
    image_embeds = extract_tensor(image_embeds)
    text_embeds = extract_tensor(text_embeds)

    # Normalize features
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # Cosine similarity matrix scaled by temperature
    logits_per_image = logit_scale * (image_embeds @ text_embeds.T)
    logits_per_text = logits_per_image.T

    labels = torch.arange(len(image_embeds), device=image_embeds.device)
    loss_img = F.cross_entropy(logits_per_image, labels)
    loss_txt = F.cross_entropy(logits_per_text, labels)

    return (loss_img + loss_txt) / 2.0


def evaluate_retrieval_acc(model, processor, val_loader, device):
    """Evaluates Top-1 and Top-5 text-to-image retrieval accuracy on validation set."""
    model.eval()
    all_image_embeds = []
    all_text_embeds = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"]
            texts = batch["llava_caption"]

            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
            image_features = extract_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
            text_features = extract_tensor(model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")))

            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            all_image_embeds.append(image_features.cpu())
            all_text_embeds.append(text_features.cpu())

    if not all_image_embeds:
        return 0.0, 0.0

    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)

    # Similarity matrix: (N_query x N_image)
    sim_matrix = text_embeds @ image_embeds.T
    N = sim_matrix.size(0)
    targets = torch.arange(N)

    # Top-1 accuracy
    top1_preds = sim_matrix.argmax(dim=-1)
    top1_acc = (top1_preds == targets).float().mean().item() * 100.0

    # Top-5 accuracy
    _, top5_preds = sim_matrix.topk(k=min(5, N), dim=-1)
    top5_acc = (top5_preds == targets.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0

    return top1_acc, top5_acc


def train_clip_lora(
    manifest_path: Path,
    captions_path: Path,
    output_dir: Path,
    model_name: str = "openai/clip-vit-base-patch32",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 5e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    aux_loss_weight: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Starting CLIP LoRA Fine-Tuning on {device}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(captions_path, "r", encoding="utf-8") as f:
        captions_dict = json.load(f)

    train_items = [item for item in manifest if item["split"] == "train"]
    val_items = [item for item in manifest if item["split"] == "val"]

    print(f"Dataset summary -> Train samples: {len(train_items)} | Val samples: {len(val_items)}")

    # Load processor and base model
    processor = CLIPProcessor.from_pretrained(model_name)
    base_model = CLIPModel.from_pretrained(model_name)

    # Apply LoRA using PEFT
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA initialized! Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Data loaders
    image_root = manifest_path.parent.parent
    train_dataset = ProductDataset(train_items, captions_dict, processor, image_root)
    val_dataset = ProductDataset(val_items, captions_dict, processor, image_root)

    def collate_fn(batch):
        return {
            "image": [item["image"] for item in batch],
            "llava_caption": [item["llava_caption"] for item in batch],
            "metadata_text": [item["metadata_text"] for item in batch],
            "image_id": [item["image_id"] for item in batch]
        }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07)).to(device)

    # Setup W&B if available
    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb_run = wandb.init(project="clip-llava-retrieval", config={
                "model_name": model_name, "epochs": epochs, "batch_size": batch_size,
                "lr": lr, "lora_r": lora_r, "train_samples": len(train_items)
            })
            print("Weights & Biases logging initialized.")
        except Exception as e:
            print(f"W&B notice: {e}")

    training_logs = []
    best_val_top1 = -1.0

    print("Beginning Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        start_epoch_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar):
            images = batch["image"]
            captions = batch["llava_caption"]
            metadata = batch["metadata_text"]

            # Tokenize and encode image & text
            inputs_cap = processor(text=captions, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
            inputs_meta = processor(text=metadata, padding=True, truncation=True, return_tensors="pt").to(device)

            image_embeds = model.get_image_features(pixel_values=inputs_cap["pixel_values"])
            caption_embeds = model.get_text_features(input_ids=inputs_cap["input_ids"], attention_mask=inputs_cap.get("attention_mask"))
            metadata_embeds = model.get_text_features(input_ids=inputs_meta["input_ids"], attention_mask=inputs_meta.get("attention_mask"))

            # Primary contrastive loss (Image + LLaVA Caption)
            loss_caption = info_nce_loss(image_embeds, caption_embeds, logit_scale.exp())
            # Auxiliary contrastive loss (Image + Metadata Title)
            loss_meta = info_nce_loss(image_embeds, metadata_embeds, logit_scale.exp())

            loss = loss_caption + aux_loss_weight * loss_meta

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / max(1, len(train_loader))
        epoch_time = time.time() - start_epoch_time

        # Validation Evaluation
        val_top1, val_top5 = evaluate_retrieval_acc(model, processor, val_loader, device)
        print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | Val Top-1: {val_top1:.2f}% | Val Top-5: {val_top5:.2f}% | Time: {epoch_time:.1f}s")

        epoch_log = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_top1_acc": val_top1,
            "val_top5_acc": val_top5,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time
        }
        training_logs.append(epoch_log)

        if wandb_run:
            wandb_run.log(epoch_log)

        # Save checkpoint if best
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            checkpoint_dir = output_dir / "best_model"
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            print(f"Saved new best model checkpoint to {checkpoint_dir} (Val Top-1: {val_top1:.2f}%)")

    # Save training log JSON
    log_file = output_dir / "training_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(training_logs, f, indent=2)

    print(f"Fine-tuning complete! Training logs saved to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Fine-Tuning Stage")
    parser.add_argument("--manifest", type=str, default="data/manifest.json", help="Path to manifest JSON")
    parser.add_argument("--captions", type=str, default="data/captions.json", help="Path to captions JSON cache")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="Pretrained CLIP model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    train_clip_lora(
        manifest_path=Path(args.manifest),
        captions_path=Path(args.captions),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
