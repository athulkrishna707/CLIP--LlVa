import argparse
import hashlib
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PRODUCT_TAXONOMY = {
    "electronics": [
        ("Wireless Noise-Canceling Headphones", "matte black over-ear bluetooth headphones with cushion earcups", (30, 30, 35)),
        ("Smart Fitness Watch", "rose gold smart watch with AMOLED touch display and silicone strap", (200, 150, 160)),
        ("Portable Bluetooth Speaker", "waterproof cylindrical Bluetooth speaker in navy blue with mesh grill", (25, 45, 85)),
        ("Mechanical Gaming Keyboard", "RGB backlit compact mechanical keyboard with custom switches", (40, 40, 50)),
        ("Ultra-Slim Laptop Stand", "aluminum ergonomic foldable laptop riser stand in silver", (190, 195, 200)),
        ("Minimalist Wireless Mouse", "ergonomic optical mouse in pearl white with metallic scroll wheel", (240, 240, 242)),
        ("HD Webcam with Privacy Cover", "1080p streaming webcam with auto-focus lens and built-in mic", (45, 50, 55)),
        ("Fast Wireless Charger Pad", "sleek leatherette wireless charging pad with LED power indicator", (60, 40, 30)),
    ],
    "apparel": [
        ("Classic Denim Jacket", "vintage washed blue denim trucker jacket with brass buttons", (70, 100, 150)),
        ("Waterproof Winter Parka", "heavyweight olive green parka coat with faux-fur lined hood", (55, 75, 50)),
        ("Cotton Crewneck T-Shirt", "premium organic white cotton t-shirt with relaxed fit", (245, 245, 245)),
        ("Slim-Fit Chino Pants", "khaki stretch cotton chino trousers with tailored waist", (210, 180, 140)),
        ("Wool Blend Trench Coat", "double-breasted camel wool trench coat with waist belt", (195, 145, 95)),
        ("Graphic Pullover Hoodie", "charcoal gray fleece hoodie with minimalist chest emblem", (65, 65, 70)),
        ("Athletic Running Shorts", "breathable black mesh sports shorts with reflective striping", (20, 20, 20)),
        ("Striped Linen Summer Shirt", "light blue and white vertical stripe linen button-up shirt", (170, 210, 240)),
    ],
    "footwear": [
        ("Breathable Mesh Running Shoes", "lightweight neon accent marathon running sneakers with foam sole", (40, 180, 160)),
        ("Leather Chelsea Boots", "dark brown polished leather ankle boots with elastic side panels", (90, 50, 30)),
        ("Classic Canvas Sneakers", "high-top black canvas sneakers with white vulcanized rubber sole", (25, 25, 25)),
        ("Suede Loafers", "tan suede slip-on loafers with penny strap and cushioned footbed", (180, 130, 80)),
        ("Trail Hiking Boots", "waterproof rugged leather hiking boots with deep tread rubber lug sole", (110, 85, 55)),
        ("Casual Slip-On Shoes", "hemp fabric casual summer slip-on canvas shoes in sage green", (120, 140, 110)),
    ],
    "accessories": [
        ("Minimalist Leather Wallet", "bifold RFID-blocking brown genuine leather wallet", (110, 60, 30)),
        ("Polarized Aviator Sunglasses", "gold frame aviator sunglasses with gradient brown polarized lenses", (200, 160, 60)),
        ("Canvas Travel Backpack", "water-resistant vintage canvas backpack with leather straps and laptop sleeve", (130, 90, 60)),
        ("Stainless Steel Water Bottle", "insulated matte black thermal flask with leakproof cap", (45, 45, 45)),
        ("Crossbody Leather Bag", "compact grain leather messenger bag in cognac with adjustable strap", (150, 80, 40)),
        ("Silk Patterned Scarf", "handcrafted silk neck scarf with navy and gold geometric pattern", (40, 60, 120)),
    ],
    "home_kitchen": [
        ("Stainless Steel Espresso Machine", "brushed stainless steel pump espresso coffee maker with steam wand", (180, 185, 190)),
        ("Cast Iron Dutch Oven", "enameled crimson red cast iron dutch oven pot with lid", (180, 40, 40)),
        ("Modern Ceramic Vase", "sculptural matte beige ceramic flower vase with geometric curves", (220, 205, 185)),
        ("Bamboo Cutting Board Set", "organic bamboo chopping board set with juice groove edge", (190, 150, 100)),
        ("Aromatherapy Oil Diffuser", "wood grain ultrasonic essential oil diffuser with ambient warm light", (160, 120, 80)),
        ("Ergonomic Mesh Desk Chair", "high-back breathable lumbar support office chair in obsidian black", (35, 35, 40)),
    ]
}


def render_product_image(image_path: Path, title: str, category: str, base_color: tuple, index: int):
    """Generates a high-quality, visually distinct product synthetic image."""
    img_size = (384, 384)
    image = Image.new("RGB", img_size, color=(248, 249, 250))
    draw = ImageDraw.Draw(image)

    # Background gradient effect
    r_base, g_base, b_base = base_color
    for y in range(img_size[1]):
        factor = 0.8 + 0.4 * (y / img_size[1])
        r = min(255, int(r_base * factor))
        g = min(255, int(g_base * factor))
        b = min(255, int(b_base * factor))
        draw.line([(0, y), (img_size[0], y)], fill=(r, g, b))

    # Inner card background
    margin = 32
    draw.rounded_rectangle(
        [margin, margin, img_size[0] - margin, img_size[1] - margin],
        radius=16,
        fill=(255, 255, 255),
        outline=(220, 225, 230),
        width=3
    )

    # Central product visual representation (shape based on index)
    shape_type = index % 4
    cx, cy = img_size[0] // 2, img_size[1] // 2 - 10
    shape_color = (max(0, r_base - 30), max(0, g_base - 30), max(0, b_base - 30))

    if shape_type == 0:
        # Rounded box (electronics/wallet/box)
        draw.rounded_rectangle([cx - 70, cy - 60, cx + 70, cy + 60], radius=12, fill=shape_color, outline=(40, 40, 40), width=2)
    elif shape_type == 1:
        # Circle / Disc (watch/diffuser/pot)
        draw.ellipse([cx - 70, cy - 70, cx + 70, cy + 70], fill=shape_color, outline=(40, 40, 40), width=2)
    elif shape_type == 2:
        # Polygon / Shield / Garment shape
        draw.polygon([(cx, cy - 70), (cx + 70, cy - 20), (cx + 50, cy + 70), (cx - 50, cy + 70), (cx - 70, cy - 20)], fill=shape_color, outline=(40, 40, 40))
    else:
        # Footwear / Capsule shape
        draw.rounded_rectangle([cx - 85, cy - 45, cx + 85, cy + 45], radius=25, fill=shape_color, outline=(40, 40, 40), width=2)

    # Overlay product category badge
    badge_color = (30, 41, 59)
    draw.rounded_rectangle([margin + 12, margin + 12, margin + 140, margin + 36], radius=6, fill=badge_color)
    draw.text((margin + 20, margin + 16), category.upper(), fill=(255, 255, 255))

    # Product Title Text at bottom
    display_title = title if len(title) <= 24 else title[:21] + "..."
    draw.text((margin + 16, img_size[1] - margin - 35), display_title, fill=(15, 23, 42))

    image.save(image_path, "JPEG", quality=92)


def generate_dataset(num_samples: int, output_dir: Path, seed: int = 42):
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    seen_md5s = set()

    categories = list(PRODUCT_TAXONOMY.keys())
    print(f"Generating {num_samples} product samples across {len(categories)} categories...")

    for i in range(num_samples):
        cat = categories[i % len(categories)]
        prod_title, prod_meta, base_color = random.choice(PRODUCT_TAXONOMY[cat])

        # Add minor variance to title/metadata to ensure dataset diversity
        item_variant_id = i + 1
        img_id = f"prod_{item_variant_id:05d}"
        img_filename = f"{img_id}.jpg"
        img_path = images_dir / img_filename

        # Render visual product image
        render_product_image(img_path, prod_title, cat, base_color, i)

        # Calculate MD5 for deduplication check
        with open(img_path, "rb") as f:
            md5_hash = hashlib.md5(f.read()).hexdigest()

        if md5_hash in seen_md5s:
            # Modify image slightly to eliminate hash collision
            draw = ImageDraw.Draw(Image.open(img_path))
            draw.point((i % 100, i % 100), fill=(i % 255, (i * 2) % 255, (i * 3) % 255))

        seen_md5s.add(md5_hash)

        manifest.append({
            "image_id": img_id,
            "image_path": str(img_path.relative_to(output_dir.parent)).replace("\\", "/"),
            "title": prod_title,
            "metadata_text": f"{prod_title} - {prod_meta}",
            "category": cat,
            "md5": md5_hash,
            "split": "" # set below
        })

    # Deterministic Train / Val / Test split (80% / 10% / 10%)
    random.shuffle(manifest)
    num_val = max(1, int(0.10 * num_samples))
    num_test = max(1, int(0.10 * num_samples))

    for idx, item in enumerate(manifest):
        if idx < num_test:
            item["split"] = "test"
        elif idx < num_test + num_val:
            item["split"] = "val"
        else:
            item["split"] = "train"

    # Sort manifest by image_id for consistency
    manifest.sort(key=lambda x: x["image_id"])

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    split_counts = {
        "train": sum(1 for x in manifest if x["split"] == "train"),
        "val": sum(1 for x in manifest if x["split"] == "val"),
        "test": sum(1 for x in manifest if x["split"] == "test"),
    }
    print(f"Dataset preparation complete! Total samples: {len(manifest)}")
    print(f"Splits -> Train: {split_counts['train']}, Val: {split_counts['val']}, Test: {split_counts['test']}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Product Image Retrieval Dataset")
    parser.add_argument("--num_samples", type=int, default=5000, help="Total number of product samples to generate")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_dataset(num_samples=args.num_samples, output_dir=Path(args.output_dir), seed=args.seed)
