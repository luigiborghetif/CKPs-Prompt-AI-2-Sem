from zipfile import ZipFile

import pytest

from scripts.package_ckp01 import package


def test_package_excludes_env_and_caches(tmp_path):
    (tmp_path / "app" / "__pycache__").mkdir(parents=True)
    (tmp_path / "README.md").write_text("instruções")
    (tmp_path / ".env.example").write_text("OLLAMA_API_KEY=cole_sua_chave_aqui")
    (tmp_path / ".env").write_text("OLLAMA_API_KEY=private-value")
    (tmp_path / "app" / "main.py").write_text("pass")
    (tmp_path / "app" / "__pycache__" / "secret.py").write_text("private-value")
    (tmp_path / "app" / ".env").write_text("private-value")
    with ZipFile(package(tmp_path)) as archive:
        assert set(archive.namelist()) == {"README.md", ".env.example", "app/main.py"}


def test_package_rejects_key_in_evidence(tmp_path):
    (tmp_path / "evidence/ckp01").mkdir(parents=True)
    (tmp_path / ".env").write_text("OLLAMA_API_KEY=private-value")
    (tmp_path / "evidence/ckp01/results.json").write_text('"private-value"')
    with pytest.raises(ValueError, match="Credencial detectada"):
        package(tmp_path)
