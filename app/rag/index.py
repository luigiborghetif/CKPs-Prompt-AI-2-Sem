"""Indexação opcional: python -m app.rag.index, após configurar embeddings."""

import json

from app.rag.pipeline import default_pipeline
from app.service import public_error


def main():
    try:
        print(json.dumps(default_pipeline().index(), ensure_ascii=False, indent=2))
    except Exception as exc:
        raise SystemExit(public_error(exc)) from None


if __name__ == "__main__":
    main()
