"""Entrega CKP02 com notebook e fontes, sem índice/segredos/cache."""

import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "README.md",
    "PLAN.md",
    ".env.example",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-rag.txt",
    "pyproject.toml",
    "LICENSE",
}
TREES = {"app", "data", "evals", "tests", "scripts", "notebooks", "evidence"}
EXTENSIONS = {".py", ".md", ".json", ".csv", ".png", ".tiktoken", ".txt", ".html", ".pdf", ".ipynb"}


def package(root: Path = ROOT) -> Path:
    files = [root / name for name in FILES if (root / name).is_file()]
    for tree in TREES:
        if not (root / tree).is_dir():
            continue
        for path in (root / tree).rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or (path.suffix not in EXTENSIONS and path.name != "LICENSE"):
                continue
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Não empacotar arquivos externos ligados por symlink.")
            files.append(path)
    key = os.environ.get("OLLAMA_API_KEY") or dotenv_values(root / ".env").get("OLLAMA_API_KEY")
    if key and key not in {"cole_sua_chave_aqui", "sua_chave_aqui"}:
        for path in files:
            if key.encode() in path.read_bytes():
                raise ValueError(f"Credencial detectada em {path.relative_to(root)}.")
    destination = root / "dist/CKP02_residuos_sp_grupo.zip"
    destination.parent.mkdir(exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root))
    return destination


if __name__ == "__main__":
    print(f"Entrega gerada: {package()}")
