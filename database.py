import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any

DEFAULT_DB_PATH = Path("data/products.db")


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Returns a connection to the SQLite products database."""
    target_path = Path(db_path or os.getenv("PRODUCTS_DB_PATH", DEFAULT_DB_PATH))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_products_db(db_path: Optional[Path] = None) -> Path:
    """Initializes the proper relational products schema in the database."""
    target_path = Path(db_path or os.getenv("PRODUCTS_DB_PATH", DEFAULT_DB_PATH))
    conn = get_db_connection(target_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                category TEXT NOT NULL,
                brand TEXT,
                price REAL,
                image_path TEXT,
                image_url TEXT,
                split TEXT DEFAULT 'train',
                md5 TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_products_split ON products(split);
        """)
    conn.close()
    return target_path


def insert_or_update_product(product: Dict[str, Any], db_path: Optional[Path] = None):
    """Inserts or updates a product row in the database."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute("""
            INSERT INTO products (id, title, description, category, brand, price, image_path, image_url, split, md5)
            VALUES (:id, :title, :description, :category, :brand, :price, :image_path, :image_url, :split, :md5)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                description=excluded.description,
                category=excluded.category,
                brand=COALESCE(excluded.brand, products.brand),
                price=COALESCE(excluded.price, products.price),
                image_path=COALESCE(excluded.image_path, products.image_path),
                image_url=COALESCE(excluded.image_url, products.image_url),
                split=COALESCE(excluded.split, products.split),
                md5=COALESCE(excluded.md5, products.md5),
                updated_at=CURRENT_TIMESTAMP
        """, {
            "id": product["id"],
            "title": product["title"],
            "description": product.get("description", ""),
            "category": product.get("category", "general"),
            "brand": product.get("brand"),
            "price": product.get("price"),
            "image_path": product.get("image_path"),
            "image_url": product.get("image_url"),
            "split": product.get("split", "train"),
            "md5": product.get("md5")
        })
    conn.close()


def fetch_all_products(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieves all product records from the database ordered by id."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, title, description, category, brand, price, image_path, image_url, split, md5, created_at
        FROM products
        ORDER BY id ASC
    """)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_products_count(db_path: Optional[Path] = None) -> int:
    """Returns the total number of products in the database."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM products")
    count = cursor.fetchone()[0]
    conn.close()
    return count

