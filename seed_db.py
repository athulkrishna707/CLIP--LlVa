import json
from pathlib import Path
from database import init_products_db, insert_or_update_product, get_products_count

def seed_database():
    db_path = init_products_db()
    manifest_file = Path("data/manifest.json")
    
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            items = json.load(f)
            
        print(f"Migrating {len(items)} product records into SQL database '{db_path}'...")
        for item in items:
            product = {
                "id": item.get("id") or item.get("image_id"),
                "title": item.get("title", ""),
                "description": item.get("description", item.get("metadata_text", "")),
                "category": item.get("category", "general"),
                "brand": item.get("brand", "Standard"),
                "price": item.get("price", 49.99),
                "image_path": item.get("image_path", ""),
                "image_url": item.get("image_url", ""),
                "split": item.get("split", "train"),
                "md5": item.get("md5", "")
            }
            insert_or_update_product(product, db_path=db_path)
            
    count = get_products_count(db_path)
    print(f"Database successfully populated! Total records in SQL database: {count}")

if __name__ == "__main__":
    seed_database()

