"""Load ontology and lookup tables."""
# stdlib
import json
import os.path as op
from pathlib import Path

# externals
import numpy as np
import yaml

PATH_ALLEN = op.join(op.dirname(__file__), "lut", "AllenBrainOntologyDev.json")
PATH_NEXTBRAIN = op.join(op.dirname(__file__), "lut", "NextBrainLUT.txt")

PathLike = str | Path
ColorRGB = tuple[int, int, int]
ColorRGBA = tuple[int, int, int, int]
Color = ColorRGB | ColorRGBA

LUT_DTYPE = np.dtype([
    ("ID", "i8"),
    ("NAME", "S256"),
    ("R", "u8"),
    ("G", "u8"),
    ("B", "u8"),
    ("A", "u8"),
])


def load_yaml(fname: PathLike) -> dict:
    """Load a YAML file."""
    with open(fname) as f:
        data = yaml.safe_load(f)
    return data


def load_json(fname: PathLike) -> dict:
    """Load a JSON file."""
    with open(fname) as f:
        data = json.load(f)
    return data


def load_ontology(fname: PathLike = PATH_ALLEN) -> np.ndarray:
    """Load the allen ontology in JSON format."""
    return load_json(fname)


def load_lut(fname: PathLike = PATH_NEXTBRAIN) -> "LUT":
    """Load the nextbrain lookup in numpy format."""
    text = []
    with open(fname) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            label, name, *color = line.split()
            label, *color = map(int, (label, *color))
            text.append((label, name, *color[:4]))

    lut = np.ndarray([len(text)], dtype=LUT_DTYPE)
    for i, line in enumerate(text):
        lut[i] = line
    return lut.view(LUT)


class LUT(np.ndarray):
    """Freesurfer lookup table."""

    @classmethod
    def load(cls, fname: PathLike = PATH_NEXTBRAIN) -> "LUT":
        """Load a lookup table from file."""
        return load_lut(fname)

    @property
    def labels(self) -> list[int]:
        """Get label IDs."""
        return self["ID"].tolist()

    @property
    def names(self) -> list[str]:
        """Get label names."""
        return [name.decode("utf-8") for name in self["NAME"].tolist()]

    @property
    def colors(self) -> np.ndarray:
        """Get label colors."""
        return self[["R", "G", "B", "A"]].astype("u1")

    def label2name(self, label: int | None = None) -> dict[int, str] | str:
        """Get mapping from label IDs to names."""
        if label is not None:
            return self[self["ID"] == label]["NAME"][0].decode("utf-8")

        return {
            int(lab): name.decode("utf-8")
            for lab, name in zip(self["ID"], self["NAME"])
        }

    def name2label(self, name: str | None = None) -> dict[str, int] | int:
        """Get mapping from label names to IDs."""
        if name is not None:
            return int(self[self["NAME"] == name.encode("utf-8")]["ID"][0])

        return {
            name.decode("utf-8"): int(lab)
            for lab, name in zip(self["ID"], self["NAME"])
        }

    def label2color(
        self, label: int | None = None
    ) -> dict[int, ColorRGBA] | ColorRGBA:
        """Get mapping from label IDs to colors."""
        if label is not None:
            color = (
                self[self["ID"] == label][["R", "G", "B", "A"]]
                .astype("u1")[0]
            )
            return tuple(color)
        return {
            int(lab): tuple(color)
            for lab, color in zip(self.labels, self.colors)
        }

    def name2color(
        self, name: str | None = None
    ) -> dict[str, ColorRGBA] | ColorRGBA:
        """Get mapping from label names to colors."""
        if name is not None:
            color = self[
                self["NAME"] == name.encode("utf-8")
            ][["R", "G", "B", "A"]].astype("u1")[0]
            return tuple(color)

        return {
            name: tuple(color)
            for name, color in zip(self.names, self.colors)
        }
