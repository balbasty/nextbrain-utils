"""Generate Freesurfer lookup tables."""
__author__ = "Yael Balbastre"

# std
import json
import os.path as op
from pathlib import Path
from typing import Iterator, TextIO

PATH_ALLEN = op.join(op.dirname(__file__), "lut", "AllenBrainOntologyDev.json")
PATH_NEXTBRAIN = op.join(op.dirname(__file__), "lut", "NextBrainLUT.txt")

PathLike = str | Path


def allen_lut(
    acronym: bool = False,
    append_dk: bool = False,
    save: TextIO | PathLike | bool = False,
) -> list[tuple[int, str, int, int, int, int]]:
    """
    Compute a Freesurfer lookup table (LUT) that contains the labels
    and colors form the Allen Brain ontology.

    Parameters
    ----------
    acronym : bool, default=False
        If True, use acronyms instead of full names for the label names.
    append_dk : bool, default=False
        If True, append Desikan-Killiany labels to the LUT.
    save:  PathLike | bool = True
        Whether to save the LUT to disk.

    Returns
    -------
    lut : list of tuples
        The lookup table as a list of tuples
        (label, name, R, G, B, A).
    """
    if save:
        if save is True:
            save = "AllenBrainLUT.txt"
        if not hasattr(save, "write"):
            with open(save, "w") as fileobj:
                return allen_lut(acronym, append_dk, fileobj)

    lut = []
    for line in make_allen_lut(acronym=acronym):
        lut.append(line)
        if save:
            save.write(_line2str(*line) + "\n")

    if append_dk:
        for line in make_dk_lut():
            lut.append(line)
            if save:
                save.write(_line2str(*line) + "\n")

    return lut


def make_allen_lut(
    fname: PathLike = PATH_ALLEN, acronym: bool = False
) -> Iterator[tuple[int, str, int, int, int, int]]:
    """Generate lines of a freesurfer lookup table from the Allen ontology."""
    with open(fname) as f:
        allen_ontology = json.load(f)

    def recurse(ont: dict) -> Iterator[list[int | str]]:
        name = ont["acronym" if acronym else "name"]
        name = name.replace(" ", "_")
        label = ont["id"]
        r, g, b = _hex2rgb(ont["color_hex_triplet"])
        yield (label, name, r, g, b, 0)
        for child in ont.get("children", []):
            yield from recurse(child)

    yield from recurse(allen_ontology)


def make_dk_lut(
    fname: PathLike = PATH_NEXTBRAIN
) -> Iterator[tuple[int, str, int, int, int, int]]:
    """
    Generate lines of a freesurfer lookup table containing only
    Desikan-Killiany labels.
    """
    with open(fname) as f:
        for line in f:
            line = line.strip().split("#")[0].strip()
            if not f:
                continue
            label, name, r, g, b, a = line.split()
            if name.startswith("ctx-rh-"):
                name = "ctx-" + name[7:]
                label = int(label)
                r, g, b, a = map(int, (r, g, b, a))
                yield (label, name, r, g, b, a)


def _hex2rgb(hex: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    r, g, b = hex[:2], hex[2:4], hex[4:6]
    r, g, b = int(r, 16), int(g, 16), int(b, 16)
    return (r, g, b)


def _line2str(label: int, name: str, r: int, g: int, b: int, a: int) -> str:
    """Convert LUT row to string."""
    return f"{label:10d}  {name:<70s}  {r:3d} {g:3d} {b:3d} {a:3d}"
