import argparse
import hashlib
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io
import requests

PRODUCT_DATASET_SPECS = [
    # Footwear
    {
        "id": "PROD_00001",
        "title": "Neon Yellow Mesh Running Shoes",
        "category": "footwear",
        "description": "High-performance marathon running shoes crafted with breathable neon yellow mesh uppers and responsive white foam soles. Engineered for maximum airflow and cushioned impact absorption on long trail runs.",
        "bg_color": (242, 240, 241),
        "primary_color": (234, 245, 39),
        "accent_color": (30, 41, 59),
        "shape": "shoe"
    },
    {
        "id": "PROD_00002",
        "title": "Retro Low-Top Canvas Sneakers",
        "category": "footwear",
        "description": "Classic low-top canvas sneakers featuring crisp white vulcanized rubber toes and black canvas side paneling. Designed with vintage contrast stitching and durable rubber tread soles for everyday street style.",
        "bg_color": (242, 240, 241),
        "primary_color": (25, 25, 25),
        "accent_color": (240, 240, 240),
        "shape": "shoe"
    },
    {
        "id": "PROD_00003",
        "title": "Dark Brown Leather Chelsea Boots",
        "category": "footwear",
        "description": "Hand-burnished dark brown genuine leather Chelsea ankle boots with elastic black side gussets and pull tabs. Built with Goodyear welted leather soles for timeless formality and weather durability.",
        "bg_color": (242, 240, 241),
        "primary_color": (85, 45, 25),
        "accent_color": (30, 20, 15),
        "shape": "boot"
    },
    {
        "id": "PROD_00004",
        "title": "Tan Suede Penny Loafers",
        "category": "footwear",
        "description": "Refined tan suede slip-on loafers featuring a traditional penny strap across the vamp and soft leather linings. Finished with a flexible crepe rubber sole for casual elegance and lightweight comfort.",
        "bg_color": (242, 240, 241),
        "primary_color": (195, 145, 95),
        "accent_color": (150, 100, 55),
        "shape": "shoe"
    },
    {
        "id": "PROD_00005",
        "title": "Tactical Olive Trail Runners",
        "category": "footwear",
        "description": "Rugged trail running shoes built with water-repellent olive green ripstop nylon and deep lug Vibram outsoles. Reinforced with synthetic toe caps and quick-lace systems for challenging outdoor terrain.",
        "bg_color": (242, 240, 241),
        "primary_color": (75, 90, 60),
        "accent_color": (40, 50, 35),
        "shape": "shoe"
    },

    # Tops
    {
        "id": "PROD_00006",
        "title": "Vintage Black Graphic T-Shirt",
        "category": "tops",
        "description": "Heavyweight faded black cotton t-shirt featuring a distressed vintage chest graphic and dropped shoulders. Cut in a relaxed boxy fit for authentic modern streetwear layering.",
        "bg_color": (242, 240, 241),
        "primary_color": (40, 40, 45),
        "accent_color": (180, 180, 180),
        "shape": "shirt"
    },
    {
        "id": "PROD_00007",
        "title": "Cobalt Blue Fleece Pullover Hoodie",
        "category": "tops",
        "description": "Cozy cobalt blue fleece hoodie tailored with a double-lined drawstring hood and spacious kangaroo pocket. Made from heavy organic cotton fleece for exceptional warmth and relaxed daily wear.",
        "bg_color": (242, 240, 241),
        "primary_color": (25, 85, 200),
        "accent_color": (15, 55, 140),
        "shape": "shirt"
    },
    {
        "id": "PROD_00008",
        "title": "Burgundy Knitted Cotton Polo Shirt",
        "category": "tops",
        "description": "Sophisticated deep burgundy knitted polo shirt styled with a ribbed collar and mother-of-pearl buttons. Crafted from breathable fine-gauge cotton knit for a tailored smart-casual silhouette.",
        "bg_color": (242, 240, 241),
        "primary_color": (120, 25, 45),
        "accent_color": (80, 15, 30),
        "shape": "shirt"
    },
    {
        "id": "PROD_00009",
        "title": "Washed Oversized White Heavyweight Tee",
        "category": "tops",
        "description": "Clean white 260GSM combed cotton t-shirt featuring a tight crew neck collar and dropped sleeve seams. Pre-shrunk for a permanent structured drape and minimalist minimalist aesthetic.",
        "bg_color": (242, 240, 241),
        "primary_color": (250, 250, 250),
        "accent_color": (210, 215, 220),
        "shape": "shirt"
    },

    # Outerwear
    {
        "id": "PROD_00010",
        "title": "Medium-Wash Classic Denim Jacket",
        "category": "outerwear",
        "description": "Iconic medium-wash indigo denim trucker jacket detailed with contrast yellow stitching and antiqued brass shank buttons. Features twin button-flap chest pockets and adjustable waist tabs.",
        "bg_color": (242, 240, 241),
        "primary_color": (65, 110, 175),
        "accent_color": (35, 70, 125),
        "shape": "jacket"
    },
    {
        "id": "PROD_00011",
        "title": "Matte Black Insulated Puffer Coat",
        "category": "outerwear",
        "description": "Ultra-warm matte black down puffer jacket constructed with windproof nylon shell and ethical 700-fill power down insulation. Includes a high stand collar, fleece-lined zip pockets, and elastic cuffs.",
        "bg_color": (242, 240, 241),
        "primary_color": (20, 20, 25),
        "accent_color": (60, 60, 65),
        "shape": "jacket"
    },
    {
        "id": "PROD_00012",
        "title": "Camel Wool Blend Trench Coat",
        "category": "outerwear",
        "description": "Tailored double-breasted camel wool trench coat styled with exaggerated notch lapels and an adjustable waist belt. Fully lined with satin fabric for elegant formal outer coverage.",
        "bg_color": (242, 240, 241),
        "primary_color": (195, 140, 80),
        "accent_color": (140, 95, 45),
        "shape": "jacket"
    },

    # Bottoms
    {
        "id": "PROD_00013",
        "title": "Washed Gray Straight-Leg Denim Jeans",
        "category": "bottoms",
        "description": "Classic 5-pocket denim jeans in a faded charcoal gray wash with authentic whiskering details. Cut in a versatile straight-leg fit from rigid 14oz ring-spun cotton denim.",
        "bg_color": (242, 240, 241),
        "primary_color": (90, 95, 100),
        "accent_color": (50, 55, 60),
        "shape": "pants"
    },
    {
        "id": "PROD_00014",
        "title": "Khaki Tactical Multi-Pocket Cargo Pants",
        "category": "bottoms",
        "description": "Durable khaki stretch cotton ripstop cargo pants featuring spacious flap utility pockets and articulated knee darts. Fitted with an elastic drawstring waistband and adjustable ankle cuffs.",
        "bg_color": (242, 240, 241),
        "primary_color": (205, 175, 130),
        "accent_color": (150, 120, 80),
        "shape": "pants"
    },
    {
        "id": "PROD_00015",
        "title": "Athletic Black Performance Sweatpants",
        "category": "bottoms",
        "description": "Streamlined obsidian black jogger sweatpants knit from moisture-wicking technical fleece with zipped side pockets. Designed with tapered leg ribs for flexible athletic training.",
        "bg_color": (242, 240, 241),
        "primary_color": (30, 30, 35),
        "accent_color": (60, 60, 65),
        "shape": "pants"
    },

    # Accessories
    {
        "id": "PROD_00016",
        "title": "Waterproof Black Roll-Top Backpack",
        "category": "accessories",
        "description": "Minimalist matte black waterproof roll-top backpack engineered with TPU-coated nylon and sealed seam zippers. Features an internal padded 16-inch laptop sleeve and ergonomic mesh shoulder straps.",
        "bg_color": (242, 240, 241),
        "primary_color": (25, 25, 30),
        "accent_color": (70, 70, 75),
        "shape": "box"
    },
    {
        "id": "PROD_00017",
        "title": "Forest Green Cotton Baseball Cap",
        "category": "accessories",
        "description": "Unstructured 6-panel baseball cap crafted from washed forest green cotton twill with an embroidered tonal eyelet pattern. Finished with an adjustable metal strap buckle at the back.",
        "bg_color": (242, 240, 241),
        "primary_color": (35, 75, 50),
        "accent_color": (20, 45, 30),
        "shape": "circle"
    },
    {
        "id": "PROD_00018",
        "title": "Gold Aviator Polarized Sunglasses",
        "category": "accessories",
        "description": "Timeless metal frame aviator sunglasses in polished gold with dark brown glare-reducing polarized lenses. Fitted with soft silicone nose pads and UV400 protective coatings.",
        "bg_color": (242, 240, 241),
        "primary_color": (215, 175, 70),
        "accent_color": (120, 80, 30),
        "shape": "circle"
    },
    {
        "id": "PROD_00019",
        "title": "Bifold Cognac Leather Wallet",
        "category": "accessories",
        "description": "Compact cognac brown vegetable-tanned leather bifold wallet featuring 6 card slots and a bill compartment. Hand-stitched with waxed thread for lifetime durability and rich patina.",
        "bg_color": (242, 240, 241),
        "primary_color": (155, 85, 35),
        "accent_color": (100, 50, 20),
        "shape": "box"
    },
    {
        "id": "PROD_00020",
        "title": "Matte Black Wireless Over-Ear Headphones",
        "category": "accessories",
        "description": "Premium matte black wireless Bluetooth headphones engineered with active noise cancellation and memory foam ear cushions. Delivers deep bass clarity and 40-hour battery life.",
        "bg_color": (242, 240, 241),
        "primary_color": (35, 35, 40),
        "accent_color": (80, 80, 90),
        "shape": "circle"
    },
    {
        "id": "PROD_00021",
        "title": "Rose Gold AMOLED Smartwatch",
        "category": "accessories",
        "description": "Sleek rose gold aluminum smartwatch featuring a vibrant touch AMOLED display and breathable silicone wristband. Tracks continuous heart rate, sleep metrics, and GPS workouts.",
        "bg_color": (242, 240, 241),
        "primary_color": (205, 145, 155),
        "accent_color": (150, 95, 105),
        "shape": "circle"
    },

    # Home & Kitchen
    {
        "id": "PROD_00022",
        "title": "Stainless Steel Pump Espresso Machine",
        "category": "home_kitchen",
        "description": "Commercial-grade brushed stainless steel pump espresso machine with a 15-bar Italian pressure pump and steam milk frothing wand. Designed for brewing rich cremas at home.",
        "bg_color": (242, 240, 241),
        "primary_color": (185, 190, 195),
        "accent_color": (100, 105, 110),
        "shape": "box"
    },
    {
        "id": "PROD_00023",
        "title": "Crimson Red Cast Iron Dutch Oven",
        "category": "home_kitchen",
        "description": "Heavy 5.5-quart enameled cast iron Dutch oven pot coated in glossy crimson red enamel with a stainless steel lid knob. Superior heat retention ideal for slow braising and bread baking.",
        "bg_color": (242, 240, 241),
        "primary_color": (180, 35, 35),
        "accent_color": (110, 15, 15),
        "shape": "circle"
    },
    {
        "id": "PROD_00024",
        "title": "Sculptural Matte Beige Ceramic Vase",
        "category": "home_kitchen",
        "description": "Handcrafted minimalist ceramic flower vase rendered in a textured matte beige glaze with curved geometric contours. Serves as a decorative tabletop centerpiece for dry floral arrangements.",
        "bg_color": (242, 240, 241),
        "primary_color": (220, 205, 185),
        "accent_color": (170, 155, 135),
        "shape": "circle"
    },
    {
        "id": "PROD_00025",
        "title": "Organic Bamboo Juice-Groove Cutting Board",
        "category": "home_kitchen",
        "description": "Thick end-grain organic bamboo chopping board designed with deep perimeter juice grooves and side handle cutouts. Naturally antibacterial surface gentle on kitchen knife edges.",
        "bg_color": (242, 240, 241),
        "primary_color": (195, 155, 105),
        "accent_color": (145, 105, 60),
        "shape": "box"
    }
]


def render_studio_product_photo(img_path: Path, spec: dict):
    """Renders a clean, high-resolution product catalog photo with solid #F2F0F1 studio backdrop."""
    img_size = (512, 512)
    # Background: #F2F0F1 (242, 240, 241)
    image = Image.new("RGB", img_size, color=spec["bg_color"])
    draw = ImageDraw.Draw(image)

    cx, cy = img_size[0] // 2, img_size[1] // 2 - 10
    shape_type = spec.get("shape", "box")
    primary_color = spec["primary_color"]
    accent_color = spec["accent_color"]

    # Subtle soft drop shadow underneath product
    shadow_color = (215, 212, 213)
    draw.ellipse([cx - 140, cy + 90, cx + 140, cy + 130], fill=shadow_color)

    if shape_type == "shoe":
        # Athletic sneaker silhouette
        draw.rounded_rectangle([cx - 140, cy - 30, cx + 140, cy + 60], radius=35, fill=primary_color, outline=accent_color, width=4)
        draw.polygon([(cx - 70, cy - 30), (cx + 30, cy - 90), (cx + 120, cy - 30)], fill=primary_color, outline=accent_color)
        # White foam sole base
        draw.rounded_rectangle([cx - 145, cy + 40, cx + 145, cy + 85], radius=15, fill=(255, 255, 255), outline=(200, 200, 200), width=2)
    elif shape_type == "boot":
        # Chelsea boot silhouette
        draw.rounded_rectangle([cx - 100, cy - 110, cx + 90, cy + 70], radius=25, fill=primary_color, outline=accent_color, width=4)
        # Elastic side panel
        draw.polygon([(cx - 20, cy - 80), (cx + 20, cy - 80), (cx + 35, cy + 20), (cx - 35, cy + 20)], fill=accent_color)
        draw.rounded_rectangle([cx - 105, cy + 50, cx + 105, cy + 80], radius=8, fill=(30, 25, 20))
    elif shape_type == "shirt":
        # T-Shirt / Hoodie silhouette
        draw.polygon([
            (cx - 50, cy - 130), (cx + 50, cy - 130),
            (cx + 150, cy - 60), (cx + 110, cy - 10),
            (cx + 80, cy - 30), (cx + 80, cy + 120),
            (cx - 80, cy + 120), (cx - 80, cy - 30),
            (cx - 110, cy - 10), (cx - 150, cy - 60)
        ], fill=primary_color, outline=accent_color)
    elif shape_type == "jacket":
        # Outerwear Jacket silhouette with collar
        draw.rounded_rectangle([cx - 110, cy - 100, cx + 110, cy + 120], radius=20, fill=primary_color, outline=accent_color, width=4)
        draw.polygon([(cx - 50, cy - 100), (cx, cy - 30), (cx + 50, cy - 100)], fill=accent_color)
        draw.line([(cx, cy - 30), (cx, cy + 120)], fill=accent_color, width=4)
    elif shape_type == "pants":
        # Trousers / Cargo pants silhouette
        draw.polygon([
            (cx - 90, cy - 130), (cx + 90, cy - 130),
            (cx + 80, cy + 130), (cx + 10, cy + 130),
            (cx, cy - 20),
            (cx - 10, cy + 130), (cx - 80, cy + 130)
        ], fill=primary_color, outline=accent_color)
    elif shape_type == "circle":
        # Circular accessories (watch/hat/sunglasses/vase)
        draw.ellipse([cx - 110, cy - 110, cx + 110, cy + 110], fill=primary_color, outline=accent_color, width=5)
        draw.ellipse([cx - 60, cy - 60, cx + 60, cy + 60], fill=accent_color)
    else:
        # Box shape (backpack / espresso machine / wallet / cutting board)
        draw.rounded_rectangle([cx - 110, cy - 100, cx + 110, cy + 100], radius=22, fill=primary_color, outline=accent_color, width=4)
        draw.rounded_rectangle([cx - 70, cy - 60, cx + 70, cy + 60], radius=12, fill=accent_color)

    # Category and ID tag badge overlay
    tag_bg = (0, 0, 0)
    draw.rounded_rectangle([30, 30, 180, 62], radius=30, fill=tag_bg)
    draw.text((45, 38), spec["id"], fill=(255, 255, 255))

    image.save(img_path, "JPEG", quality=95)


UNSPLASH_URLS = {
    "PROD_00001": "https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa?q=80&w=600&auto=format&fit=crop",
    "PROD_00002": "https://images.unsplash.com/photo-1549298916-b41d501d3772?q=80&w=600&auto=format&fit=crop",
    "PROD_00003": "https://images.unsplash.com/photo-1608256246200-53e635b5b65f?q=80&w=600&auto=format&fit=crop",
    "PROD_00004": "https://images.unsplash.com/photo-1533867617858-e7b97e060509?q=80&w=600&auto=format&fit=crop",
    "PROD_00005": "https://images.unsplash.com/photo-1582588678413-dbf45f4823e9?q=80&w=600&auto=format&fit=crop",
    "PROD_00006": "https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?q=80&w=600&auto=format&fit=crop",
    "PROD_00007": "https://images.unsplash.com/photo-1556821840-3a63f95609a7?q=80&w=600&auto=format&fit=crop",
    "PROD_00008": "https://images.unsplash.com/photo-1581655353564-df123a1eb820?q=80&w=600&auto=format&fit=crop",
    "PROD_00009": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?q=80&w=600&auto=format&fit=crop",
    "PROD_00010": "https://images.unsplash.com/photo-1576871337632-b9aef4c17ab9?q=80&w=600&auto=format&fit=crop",
    "PROD_00011": "https://images.unsplash.com/photo-1544923246-77307dd654cb?q=80&w=600&auto=format&fit=crop",
    "PROD_00012": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?q=80&w=600&auto=format&fit=crop",
    "PROD_00013": "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?q=80&w=600&auto=format&fit=crop",
    "PROD_00014": "https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?q=80&w=600&auto=format&fit=crop",
    "PROD_00015": "https://images.unsplash.com/photo-1551854838-212c50b4c184?q=80&w=600&auto=format&fit=crop",
    "PROD_00016": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?q=80&w=600&auto=format&fit=crop",
    "PROD_00017": "https://images.unsplash.com/photo-1588850561407-ed78c282e89b?q=80&w=600&auto=format&fit=crop",
    "PROD_00018": "https://images.unsplash.com/photo-1511499767150-a48a237f0083?q=80&w=600&auto=format&fit=crop",
    "PROD_00019": "https://images.unsplash.com/photo-1604176354204-9268737828e4?q=80&w=600&auto=format&fit=crop",
    "PROD_00020": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?q=80&w=600&auto=format&fit=crop",
    "PROD_00021": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?q=80&w=600&auto=format&fit=crop",
    "PROD_00022": "https://images.unsplash.com/photo-1584269600464-37b1b58a9fe7?q=80&w=600&auto=format&fit=crop",
    "PROD_00023": "https://images.unsplash.com/photo-1585238342024-78d387f4a707?q=80&w=600&auto=format&fit=crop",
    "PROD_00024": "https://images.unsplash.com/photo-1578500494198-246f612d3b3d?q=80&w=600&auto=format&fit=crop",
    "PROD_00025": "https://images.unsplash.com/photo-1563245372-f21724e3856d?q=80&w=600&auto=format&fit=crop",
}


def download_real_product_photo(img_path: Path, spec: dict):
    """Downloads a high-quality product photo from Unsplash and saves it as a 512x512 JPEG."""
    img_id = spec["id"]
    url = UNSPLASH_URLS.get(img_id)
    if not url:
        print(f"No URL found for {img_id}, falling back to rendering studio geometry.")
        render_studio_product_photo(img_path, spec)
        return

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        # Center crop to a square aspect ratio
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) / 2
        top = (height - min_dim) / 2
        right = (width + min_dim) / 2
        bottom = (height + min_dim) / 2
        image = image.crop((left, top, right, bottom))
        
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        image.save(img_path, "JPEG", quality=95)
        print(f"Successfully downloaded and saved real photo for {img_id}")
    except Exception as e:
        print(f"Error downloading photo for {img_id} ({e}), falling back to rendering studio geometry.")
        render_studio_product_photo(img_path, spec)


def generate_mock_dataset(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    print(f"Generating mock dataset with {len(PRODUCT_DATASET_SPECS)} distinct product items...")

    for idx, spec in enumerate(PRODUCT_DATASET_SPECS):
        img_id = spec["id"]
        img_filename = f"{img_id}.jpg"
        img_path = images_dir / img_filename

        # Download real photo or fallback to rendering studio geometry
        download_real_product_photo(img_path, spec)

        # Compute MD5 for checksum verification
        with open(img_path, "rb") as f:
            md5_hash = hashlib.md5(f.read()).hexdigest()

        # Split allocation: 80% train (20 items), 10% val (2 items), 10% test (3 items)
        if idx >= 22:
            split = "test"
        elif idx >= 20:
            split = "val"
        else:
            split = "train"

        manifest.append({
            "image_id": img_id,
            "id": img_id,
            "image_path": str(img_path.relative_to(output_dir.parent)).replace("\\", "/"),
            "title": spec["title"],
            "description": spec["description"],
            "metadata_text": f"{spec['title']} - {spec['description']}",
            "category": spec["category"],
            "md5": md5_hash,
            "split": split
        })

    # Save dataset_manifest.json
    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Also save standard manifest.json for compatibility
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Mock dataset successfully created!")
    print(f"Manifest written to: {manifest_path} ({len(manifest)} items).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Mock Product Dataset")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory path")
    args = parser.parse_args()

    generate_mock_dataset(output_dir=Path(args.output_dir))
