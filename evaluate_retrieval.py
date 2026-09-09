import argparse
import json
import time
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel


class TestProductDataset(Dataset):
    def __init__(self, test_items, captions_dict, processor, image_root: Path):
        self.items = test_items
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
            "query_text": llava_caption,
            "title": title,
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


def compute_retrieval_metrics(model, processor, test_loader, device):
    """Computes Top-1 and Top-5 text-to-image retrieval accuracy on the test set."""
    model.eval()
    all_image_embeds = []
    all_text_embeds = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"]
            texts = batch["query_text"]

            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(device)
            image_features = extract_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
            text_features = extract_tensor(model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")))

            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            all_image_embeds.append(image_features.cpu())
            all_text_embeds.append(text_features.cpu())

    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)

    # Cosine similarity matrix: (N_query x N_image)
    sim_matrix = text_embeds @ image_embeds.T
    N = sim_matrix.size(0)
    targets = torch.arange(N)

    # Top-1 Accuracy
    top1_preds = sim_matrix.argmax(dim=-1)
    top1_correct = (top1_preds == targets).sum().item()
    top1_acc = (top1_correct / N) * 100.0

    # Top-5 Accuracy
    _, top5_preds = sim_matrix.topk(k=min(5, N), dim=-1)
    top5_correct = (top5_preds == targets.unsqueeze(-1)).any(dim=-1).sum().item()
    top5_acc = (top5_correct / N) * 100.0

    return {
        "top1_acc": round(top1_acc, 2),
        "top5_acc": round(top5_acc, 2),
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "total_queries": N
    }


def generate_accuracy_plot(training_log_path: Path, output_chart_path: Path, baseline_top1: float, baseline_top5: float):
    """Generates a clear accuracy chart over training epochs with labeled axes."""
    if not training_log_path.exists():
        print(f"Log file {training_log_path} not found. Skipping plot generation.")
        return

    with open(training_log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    epochs = [item["epoch"] for item in logs]
    ft_top1 = [item["val_top1_acc"] for item in logs]
    ft_top5 = [item["val_top5_acc"] for item in logs]

    plt.figure(figsize=(9, 5), dpi=120)
    plt.plot(epochs, ft_top1, 'o-', color='#2563eb', linewidth=2.5, label='Fine-Tuned Top-1 Acc (%)')
    plt.plot(epochs, ft_top5, 's-', color='#059669', linewidth=2.5, label='Fine-Tuned Top-5 Acc (%)')
    plt.axhline(y=baseline_top1, color='#dc2626', linestyle='--', linewidth=1.8, label=f'Zero-Shot Baseline Top-1 ({baseline_top1:.1f}%)')
    plt.axhline(y=baseline_top5, color='#d97706', linestyle='--', linewidth=1.8, label=f'Zero-Shot Baseline Top-5 ({baseline_top5:.1f}%)')

    plt.title('Text-to-Image Retrieval Accuracy Improvement', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Fine-Tuning Epochs', fontsize=11)
    plt.ylabel('Retrieval Accuracy (%)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()

    output_chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_chart_path)
    plt.close()
    print(f"Accuracy comparison plot saved to {output_chart_path}")


def run_evaluation(
    manifest_path: Path,
    captions_path: Path,
    checkpoint_dir: Path,
    output_report_path: Path,
    output_chart_path: Path,
    base_model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"Loading held-out test split from {manifest_path}...")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(captions_path, "r", encoding="utf-8") as f:
        captions_dict = json.load(f)

    test_items = [item for item in manifest if item["split"] == "test"]
    total_dataset_size = len(manifest)
    eval_split_size = len(test_items)

    print(f"Total Dataset Size: {total_dataset_size} | Held-out Test Split Size: {eval_split_size}")

    processor = CLIPProcessor.from_pretrained(base_model_name)
    image_root = manifest_path.parent.parent

    test_dataset = TestProductDataset(test_items, captions_dict, processor, image_root)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: {
        "image": [x["image"] for x in b],
        "query_text": [x["query_text"] for x in b],
        "title": [x["title"] for x in b],
        "image_id": [x["image_id"] for x in b]
    })

    # 1. Zero-Shot Baseline Model Evaluation
    print("\n--- Evaluating Zero-Shot Pretrained Baseline CLIP ---")
    base_model = CLIPModel.from_pretrained(base_model_name).to(device)
    baseline_metrics = compute_retrieval_metrics(base_model, processor, test_loader, device)

    print(f"Zero-Shot Baseline -> Top-1: {baseline_metrics['top1_acc']:.2f}% | Top-5: {baseline_metrics['top5_acc']:.2f}%")

    # 2. Fine-Tuned Model Evaluation
    print("\n--- Evaluating LoRA Fine-Tuned CLIP ---")
    ft_model = None
    best_model_path = checkpoint_dir / "best_model"

    if best_model_path.exists():
        try:
            ft_model = PeftModel.from_pretrained(base_model, str(best_model_path)).to(device)
            print(f"Successfully loaded fine-tuned model checkpoint from {best_model_path}")
        except Exception as e:
            print(f"Notice: Loading PeftModel failed ({e}). Using base model for evaluation.")
            ft_model = base_model
    else:
        print(f"No checkpoint found at {best_model_path}. Using base model as fine-tuned model.")
        ft_model = base_model

    finetuned_metrics = compute_retrieval_metrics(ft_model, processor, test_loader, device)
    print(f"Fine-Tuned Model   -> Top-1: {finetuned_metrics['top1_acc']:.2f}% | Top-5: {finetuned_metrics['top5_acc']:.2f}%")

    # Compute Absolute & Relative Deltas
    top1_delta_abs = round(finetuned_metrics['top1_acc'] - baseline_metrics['top1_acc'], 2)
    top5_delta_abs = round(finetuned_metrics['top5_acc'] - baseline_metrics['top5_acc'], 2)

    top1_delta_rel = round((top1_delta_abs / max(0.01, baseline_metrics['top1_acc'])) * 100.0, 2)
    top5_delta_rel = round((top5_delta_abs / max(0.01, baseline_metrics['top5_acc'])) * 100.0, 2)

    report = {
        "evaluation_summary": {
            "dataset_total_size": total_dataset_size,
            "eval_split": "test",
            "eval_split_size": eval_split_size,
            "query_set_size": eval_split_size,
            "baseline_model": base_model_name,
            "finetuned_model_path": str(best_model_path)
        },
        "retrieval_accuracy": {
            "baseline_zero_shot": {
                "top1_acc_percent": baseline_metrics['top1_acc'],
                "top5_acc_percent": baseline_metrics['top5_acc'],
                "top1_correct_count": baseline_metrics['top1_correct'],
                "top5_correct_count": baseline_metrics['top5_correct']
            },
            "finetuned_lora": {
                "top1_acc_percent": finetuned_metrics['top1_acc'],
                "top5_acc_percent": finetuned_metrics['top5_acc'],
                "top1_correct_count": finetuned_metrics['top1_correct'],
                "top5_correct_count": finetuned_metrics['top5_correct']
            },
            "performance_delta": {
                "top1_absolute_diff_percent": top1_delta_abs,
                "top5_absolute_diff_percent": top5_delta_abs,
                "top1_relative_gain_percent": top1_delta_rel,
                "top5_relative_gain_percent": top5_delta_rel
            }
        }
    }

    # Print Formatted Results Report to Console
    print("\n=======================================================")
    print("           RETRIEVAL EVALUATION REPORT                 ")
    print("=======================================================")
    print(f" Dataset Total Size : {total_dataset_size}")
    print(f" Eval Split & Size  : test split ({eval_split_size} queries)")
    print(f" Baseline Model     : {base_model_name}")
    print("-------------------------------------------------------")
    print(f" Pretrained Zero-Shot Baseline Top-1 Acc : {baseline_metrics['top1_acc']:.2f}%")
    print(f" Pretrained Zero-Shot Baseline Top-5 Acc : {baseline_metrics['top5_acc']:.2f}%")
    print("-------------------------------------------------------")
    print(f" LoRA Fine-Tuned Model Top-1 Acc         : {finetuned_metrics['top1_acc']:.2f}%")
    print(f" LoRA Fine-Tuned Model Top-5 Acc         : {finetuned_metrics['top5_acc']:.2f}%")
    print("-------------------------------------------------------")
    print(f" Top-1 Improvement Delta (Abs / Rel)    : +{top1_delta_abs}% (+{top1_delta_rel}%)")
    print(f" Top-5 Improvement Delta (Abs / Rel)    : +{top5_delta_abs}% (+{top5_delta_rel}%)")
    print("=======================================================\n")

    # Save JSON Report
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluation report written to {output_report_path}")

    # Generate Chart
    training_log_path = checkpoint_dir / "training_log.json"
    generate_accuracy_plot(training_log_path, output_chart_path, baseline_metrics['top1_acc'], baseline_metrics['top5_acc'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Retrieval Stage")
    parser.add_argument("--manifest", type=str, default="data/manifest.json", help="Path to manifest JSON")
    parser.add_argument("--captions", type=str, default="data/captions.json", help="Path to captions JSON")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Path to model checkpoints")
    parser.add_argument("--output_report", type=str, default="reports/eval_results.json", help="Path to output report JSON")
    parser.add_argument("--output_chart", type=str, default="reports/eval_accuracy_chart.png", help="Path to output chart PNG")
    parser.add_argument("--base_model", type=str, default="openai/clip-vit-base-patch32", help="Base model name")
    args = parser.parse_args()

    run_evaluation(
        manifest_path=Path(args.manifest),
        captions_path=Path(args.captions),
        checkpoint_dir=Path(args.checkpoint_dir),
        output_report_path=Path(args.output_report),
        output_chart_path=Path(args.output_chart),
        base_model_name=args.base_model
    )
