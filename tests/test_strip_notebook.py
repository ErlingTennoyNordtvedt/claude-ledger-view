import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ledger_view.strip_notebook import Cell, Notebook, dumps, strip

ROOT = Path(__file__).resolve().parent.parent


def notebook() -> Notebook:
    return Notebook(
        cells=[
            Cell(cell_type="markdown", id="md", metadata={"tags": ["x"]}, source=["# Title\n"]),
            Cell(
                cell_type="code",
                id="code",
                metadata={"scrolled": True},
                source="print('hi')",
                execution_count=7,
                outputs=[{"output_type": "display_data", "data": {"image/png": "AAAA"}}],
            ),
        ],
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python", "version": "3.14.7"},
        },
        nbformat=4,
        nbformat_minor=5,
    )


def test_code_cell_outputs_and_count_cleared() -> None:
    code = strip(notebook())["cells"][1]
    assert code["outputs"] == []
    assert code["execution_count"] is None
    assert code["source"] == "print('hi')"
    assert code["id"] == "code"


def test_markdown_cell_keeps_source_and_gets_no_code_fields() -> None:
    md = strip(notebook())["cells"][0]
    assert md == {"cell_type": "markdown", "id": "md", "metadata": {}, "source": ["# Title\n"]}


@pytest.mark.parametrize("index", [0, 1])
def test_cell_metadata_cleared(index: int) -> None:
    assert strip(notebook())["cells"][index]["metadata"] == {}


def test_notebook_metadata_keeps_only_kernelspec() -> None:
    nb = strip(notebook())
    assert set(nb["metadata"]) == {"kernelspec"}
    assert (nb["nbformat"], nb["nbformat_minor"]) == (4, 5)


def test_input_not_mutated() -> None:
    nb = notebook()
    before = copy.deepcopy(nb)
    strip(nb)
    assert nb == before


def test_idempotent_through_dumps() -> None:
    once = dumps(strip(notebook()))
    assert dumps(strip(json.loads(once))) == once


def test_dumps_matches_jupyter_layout() -> None:
    text = dumps(strip(notebook()))
    assert text.endswith("}\n")
    assert text.startswith('{\n "cells": [\n')


def test_filter_reads_stdin_writes_stdout() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ledger_view.strip_notebook"],
        input=json.dumps(notebook()),
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    assert result.stdout == dumps(strip(notebook()))
