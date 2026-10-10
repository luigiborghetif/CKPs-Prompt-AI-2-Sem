"""Executa o notebook offline ou real com o Python atual, sem alterar kernel global."""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent


def validate(*, live=False, reuse_evidence=""):
    path = ROOT / "notebooks/CKP02_DocMind_SP.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    parameters = next(
        cell for cell in notebook.cells if cell.cell_type == "code" and "RUN_LIVE =" in cell.source
    )
    parameters.source = parameters.source.replace("RUN_LIVE = True", "RUN_LIVE = False")
    if live:
        parameters.source = parameters.source.replace("RUN_LIVE = False", "RUN_LIVE = True")
    if reuse_evidence:
        if not live or not (ROOT / reuse_evidence / "metadata.json").is_file():
            raise ValueError("Reutilização exige --live e uma pasta real de evidências.")
    parameters.source = re.sub(
        r"^REUSE_EVIDENCE\s*=.*$",
        lambda _: f"REUSE_EVIDENCE = {reuse_evidence!r}",
        parameters.source,
        flags=re.MULTILINE,
    )
    if "LAUNCH_UI = False" not in parameters.source:
        raise ValueError("Validação automática exige LAUNCH_UI=False.")
    with tempfile.TemporaryDirectory(prefix="ckp02-kernel-") as directory:
        import json

        kernel = Path(directory) / "kernels/ckp02"
        kernel.mkdir(parents=True)
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "CKP02 local",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        old = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = directory + (os.pathsep + old if old else "")
        try:
            NotebookClient(
                notebook,
                timeout=1800 if live else 120,
                kernel_name="ckp02",
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
        finally:
            if old is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = old
    nbformat.write(notebook, path)
    return path


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--live", action="store_true", help="Executa chamadas reais aos endpoints configurados."
    )
    cli.add_argument(
        "--reuse-evidence",
        default="",
        help="Pasta de comparação real para importar métricas auditáveis.",
    )
    args = cli.parse_args()
    print(
        f"Notebook executado {'real' if args.live else 'offline'}: {validate(live=args.live, reuse_evidence=args.reuse_evidence)}"
    )
