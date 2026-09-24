"""Caching layer for external API responses"""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar
from functools import wraps

from .config import get_settings

settings = get_settings()

T = TypeVar('T')


def _cache_key(func_name: str, *args, **kwargs) -> str:
    """Generate a cache key from function name and arguments"""
    key_data = f"{func_name}:{args}:{sorted(kwargs.items())}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def _cache_path(key: str) -> Path:
    """Get the cache file path for a key"""
    return settings.cache_dir / f"{key}.json"


def cached(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to cache function results to disk"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        key = _cache_key(func.__name__, *args, **kwargs)
        path = _cache_path(key)
        
        # Check cache
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                return data
        
        # Call function and cache result
        result = func(*args, **kwargs)
        
        # Ensure cache directory exists
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(result, f)
        
        return result
    return wrapper


def cached_async(func: Callable[..., T]) -> Callable[..., T]:
    """Async version of cached decorator"""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        key = _cache_key(func.__name__, *args, **kwargs)
        path = _cache_path(key)
        
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                return data
        
        result = await func(*args, **kwargs)
        
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(result, f)
        
        return result
    return wrapper


class RateLimiter:
    """Simple rate limiter with exponential backoff"""
    
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.min_interval = 60.0 / rpm if rpm > 0 else 0
        self.last_call = 0.0
    
    def wait(self) -> None:
        """Wait if necessary to respect rate limit"""
        if self.min_interval <= 0:
            return
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on: tuple = (Exception,)
):
    """Decorator for retry with exponential backoff"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        time.sleep(delay)
                    else:
                        raise
            raise last_exception
        return wrapper
    return decorator


async def retry_with_backoff_async(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on: tuple = (Exception,)
):
    """Async decorator for retry with exponential backoff"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        await asyncio.sleep(delay)
                    else:
                        raise
            raise last_exception
        return wrapper
    return decorator


import asyncio