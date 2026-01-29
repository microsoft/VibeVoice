"""Utility script to print all registered FastAPI/Starlette route paths.

Run this script from the `web-gui/backend` folder to list all routes
in the application (e.g., `python scripts/print_routes.py`).
"""

if __name__ == "__main__":
    from main import app

    # Print each registered route on its own line with helpful details
    for r in app.routes:
        path = getattr(r, "path", None)
        methods = getattr(r, "methods", None)
        if isinstance(methods, (set, list, tuple)):
            methods_str = ",".join(sorted(methods))
        else:
            methods_str = str(methods)
        endpoint = getattr(r, "endpoint", None)
        endpoint_name = getattr(endpoint, "__name__", None) or getattr(r, "name", None)
        print(f"{path}  methods={methods_str}  endpoint={endpoint_name}")

