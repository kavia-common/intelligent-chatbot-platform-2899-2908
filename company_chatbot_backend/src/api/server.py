"""Uvicorn server entrypoint for the Company Chatbot Backend.

This script allows running the FastAPI app with:
- host 0.0.0.0
- port from environment variable PORT (default 3001)

Usage:
    python -m src.api.server
"""
import os
import uvicorn


def main() -> None:
    """Run the FastAPI server with the configured host and port."""
    port = int(os.environ.get("PORT", "3001"))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
