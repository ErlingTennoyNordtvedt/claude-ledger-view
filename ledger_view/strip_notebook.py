"""Strip outputs and machine-specific metadata from a notebook, for committing it to git.

Used as a git clean filter (see `make setup`): reads a notebook on stdin, writes the stripped
notebook to stdout. Stdlib only, so it runs with any `python3`, not just in the dev shell.
"""

import json
import sys
from typing import Literal, NotRequired, TypedDict

type JSON = dict[str, JSON] | list[JSON] | str | int | float | bool | None


class Cell(TypedDict):
    cell_type: Literal["code", "markdown", "raw"]
    id: NotRequired[str]
    metadata: dict[str, JSON]
    source: str | list[str]
    attachments: NotRequired[dict[str, JSON]]
    outputs: NotRequired[list[JSON]]
    execution_count: NotRequired[int | None]


class Notebook(TypedDict):
    cells: list[Cell]
    metadata: dict[str, JSON]
    nbformat: int
    nbformat_minor: int


KEPT_NB_METADATA = ("kernelspec",)


def strip_cell(cell: Cell) -> Cell:
    stripped = Cell(cell_type=cell["cell_type"], metadata={}, source=cell["source"])
    if "id" in cell:
        stripped["id"] = cell["id"]
    if cell["cell_type"] == "code":
        stripped["execution_count"] = None
        stripped["outputs"] = []
    return stripped


def strip(nb: Notebook) -> Notebook:
    return Notebook(
        cells=[strip_cell(c) for c in nb["cells"]],
        metadata={k: nb["metadata"][k] for k in KEPT_NB_METADATA if k in nb["metadata"]},
        nbformat=nb["nbformat"],
        nbformat_minor=nb["nbformat_minor"],
    )


def dumps(nb: Notebook) -> str:
    """Serialise the way Jupyter does, so a stripped file diffs cleanly against a saved one."""
    return json.dumps(nb, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        with open(argv[1], encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    sys.stdout.write(dumps(strip(json.loads(text))))


if __name__ == "__main__":
    main(sys.argv)
