from zipfile import ZipFile

import pytest

from scripts.package_ckp02 import package


def test_delivery_includes_notebook_and_originals_but_not_secrets_or_index(tmp_path):
    for directory in ["data/raw", "notebooks", ".chroma", "app/__pycache__"]:
        (tmp_path / directory).mkdir(parents=True)
    (tmp_path / "data/raw/SP-001.html").write_text("norma real")
    (tmp_path / "notebooks/CKP02_DocMind_SP.ipynb").write_text("{}")
    (tmp_path / ".env").write_text("OLLAMA_API_KEY=private-value")
    (tmp_path / ".chroma/index.json").write_text("private-value")
    (tmp_path / "app/__pycache__/temp.py").write_text("private-value")
    with ZipFile(package(tmp_path)) as archive:
        assert set(archive.namelist()) == {
            "data/raw/SP-001.html",
            "notebooks/CKP02_DocMind_SP.ipynb",
        }


def test_key_in_notebook_output_blocks_delivery(tmp_path):
    (tmp_path / "notebooks").mkdir()
    (tmp_path / ".env").write_text("OLLAMA_API_KEY=private-value")
    (tmp_path / "notebooks/result.ipynb").write_text('"private-value"')
    with pytest.raises(ValueError, match="Credencial detectada"):
        package(tmp_path)
