"""Converte snapshots oficiais em recortes auditáveis. Não faz download."""

import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
ARTICLE = re.compile(r"^Art(?:igo)?\.?\s*(\d+)", re.IGNORECASE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_articles(raw: bytes, selected: list[int] | None = None) -> tuple[str, int]:
    soup = BeautifulSoup(raw, "html.parser")
    containers = soup.select(".bx-retornaLei")
    if len(containers) != 1:
        raise ValueError("Estrutura da página mudou; revise o seletor antes de extrair.")
    container = containers[0]
    revoked = container.select("s, strike, del, [style*='line-through']")
    removed = len(revoked)
    for node in revoked:
        node.decompose()
    groups, current = {}, None
    for node in container.select("div.customStyle > p"):
        text = node.get_text(" ", strip=True)
        text = text.translate({0x93: '"', 0x94: '"', 0x96: "–", 0xA0: " "})
        match = ARTICLE.match(text)
        if match:
            number = int(match.group(1))
            current = number if selected is None or number in selected else None
            if current is not None:
                groups.setdefault(current, [])
        if current is not None and text:
            if text.startswith(
                ("PREFEITURA DO", "Publicado na", "Este texto não", "FERNANDO HADDAD")
            ):
                current = None
                continue
            groups[current].append(text)
    if not groups:
        raise ValueError("Nenhum artigo extraído; não gerar documento vazio.")
    return "\n\n".join(
        f"## Art. {number}\n\n" + "\n\n".join(parts) for number, parts in groups.items()
    ), removed


def prepare(root: Path = ROOT) -> list[dict]:
    specs = json.loads((root / "data/source_specs.json").read_text(encoding="utf-8"))
    manifest = []
    for spec in specs:
        raw = root / "data/raw" / f"{spec['id']}.html"
        text, excluded = extract_articles(raw.read_bytes(), spec["articles"])
        processed = root / "data/processed" / f"{spec['id']}.md"
        processed.parent.mkdir(parents=True, exist_ok=True)
        # Fonte e data ficam nos metadados, não repetidas como ruído dentro de cada chunk.
        processed.write_text(text + "\n", encoding="utf-8")
        manifest.append(
            {key: value for key, value in spec.items() if key != "articles"}
            | {
                "raw_file": str(raw.relative_to(root)),
                "processed_file": str(processed.relative_to(root)),
                "raw_sha256": digest(raw),
                "processed_sha256": digest(processed),
                "excluded_revoked_elements": excluded,
            }
        )
    (root / "data/sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    for source in prepare():
        print(source["id"], source["processed_file"], source["excerpt"])
