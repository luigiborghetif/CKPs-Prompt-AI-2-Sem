"""Gera o ZIP por lista positiva de arquivos, sem .env, caches ou venv."""

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
    "pyproject.toml",
    "LICENSE",
}
TREES = {"app", "data", "evals", "tests", "scripts", "evidence"}
EXTENSIONS = {".py", ".md", ".json", ".csv", ".png", ".tiktoken", ".txt"}


def selected_files(root: Path = ROOT) -> list[Path]:
    paths = [root / name for name in FILES if (root / name).is_file()]
    for tree in TREES:
        directory = root / tree
        if directory.is_dir():
            for path in directory.rglob("*"):
                if not path.is_file() or (path.suffix not in EXTENSIONS and path.name != "LICENSE"):
                    continue
                relative = path.relative_to(root)
                # A evolução para CKP02 não altera as dependências da entrega CKP01.
                if tree == "app" and "rag" in relative.parts:
                    continue
                if tree == "data" and not (
                    path.name == "contexto_oficial.md" or "tokenizer" in relative.parts
                ):
                    continue
                if tree == "evals" and path.name not in {
                    "__init__.py",
                    "run_chat.py",
                    "chat_cases.json",
                }:
                    continue
                if tree == "tests" and path.name.startswith("test_rag"):
                    continue
                if tree == "scripts" and path.name not in {"__init__.py", "package_ckp01.py"}:
                    continue
                if tree == "evidence" and "ckp01" not in relative.parts:
                    continue
                if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                    continue
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError(
                        "A entrega não aceita links simbólicos para arquivos externos."
                    )
                paths.append(path)
    return sorted(paths)


def package(root: Path = ROOT) -> Path:
    destination = root / "dist" / "CKP01_residuos_sp_grupo.zip"
    destination.parent.mkdir(exist_ok=True)
    paths = selected_files(root)
    key = os.environ.get("OLLAMA_API_KEY") or dotenv_values(root / ".env").get("OLLAMA_API_KEY")
    if key and key not in {"cole_sua_chave_aqui", "sua_chave_aqui"}:
        for path in paths:
            if key.encode() in path.read_bytes():
                raise ValueError(
                    f"Credencial detectada em {path.relative_to(root)}. Corrija antes de empacotar."
                )
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(root))
    return destination


if __name__ == "__main__":
    print(f"Entrega gerada: {package()}")
