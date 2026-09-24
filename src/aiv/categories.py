"""Load and manage category configuration from YAML"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from .config import get_settings


@dataclass
class Brand:
    name: str
    aliases: List[str]


@dataclass
class Prompt:
    category: str
    intent: str
    text: str


@dataclass
class Category:
    name: str
    brands: List[Brand]
    prompts: List[Prompt]


def load_categories() -> Dict[str, Category]:
    """Load categories from config/categories.yaml"""
    config_path = Path("config/categories.yaml")
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    categories = {}
    for cat_data in data["categories"]:
        brands = [Brand(name=b["name"], aliases=b["aliases"]) for b in cat_data["brands"]]
        prompts = [Prompt(category=p["category"], intent=p["intent"], text=p["text"]) for p in cat_data["prompts"]]
        categories[cat_data["name"]] = Category(name=cat_data["name"], brands=brands, prompts=prompts)
    
    return categories


def get_all_brands() -> List[Brand]:
    """Get all brands across all categories"""
    categories = load_categories()
    all_brands = []
    for cat in categories.values():
        all_brands.extend(cat.brands)
    return all_brands


def get_brand_aliases() -> Dict[str, str]:
    """Get mapping from alias to canonical brand name"""
    aliases = {}
    for brand in get_all_brands():
        for alias in brand.aliases:
            aliases[alias.lower()] = brand.name
    return aliases


def normalize_brand(name: str) -> Optional[str]:
    """Normalize a brand name to its canonical form using aliases"""
    aliases = get_brand_aliases()
    return aliases.get(name.lower())


def get_prompts_by_category(category: str) -> List[Prompt]:
    """Get all prompts for a specific category"""
    categories = load_categories()
    return categories.get(category, Category(name=category, brands=[], prompts=[])).prompts


def get_all_prompts() -> List[Prompt]:
    """Get all prompts across all categories"""
    categories = load_categories()
    all_prompts = []
    for cat in categories.values():
        all_prompts.extend(cat.prompts)
    return all_prompts