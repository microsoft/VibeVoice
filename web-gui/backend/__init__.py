# backend package initializer
# This file makes the backend a proper package so that intra-package
# imports can use relative imports (e.g., from .config import settings).
__all__ = ["config", "routes", "main"]
