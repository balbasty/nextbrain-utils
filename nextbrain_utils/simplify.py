"""Convert NextBrain labels to Allen ontology labels."""
__author__ = "Yael Balbastre"

# std
import os.path as op
from pathlib import Path

# externals
import nibabel as nb
import numpy as np
from nibabel.spatialimages import SpatialImage
from numpy.typing import ArrayLike

# internals
from .to_allen import load_allen_ontology, normalize_name

PathLike = str | Path


def get_allen2simple_map(
    labels: list[int | str],
    hide_missing: bool = False,
) -> np.ndarray:
    """Compute linear label maps."""
    # convert integer-like labels to integers
    labels, _ = [], labels
    for label in _:
        try:
            label = int(label)
        except ValueError:
            ...
        labels.append(label)

    # load lookup tables
    allen_ont = load_allen_ontology()
    allen_dtype = 'i4'

    # prepare linear label maps
    if hide_missing:
        allen2simple = np.zeros([2**29], dtype=allen_dtype)
    else:
        allen2simple = np.arange(2**29, dtype=allen_dtype)

    # Map NextBrain labels to Allen labels
    def _recurse_map(ont: dict, target: int) -> None:
        allen2simple[ont['id']] = target
        for child in ont.get("children", []):
            _recurse_map(child, target)

    def _recurse(ont: dict) -> None:
        id = ont["id"]
        acronym = ont["acronym"]
        name = ont["name"]
        name_norm = normalize_name(name)
        if (
            (acronym in labels) or
            (name in labels) or
            (name_norm in labels) or
            (id in labels)
        ):
            print(ont["name"])
            _recurse_map(ont, ont['id'])
        else:
            for child in ont.get("children", []):
                _recurse(child)

    _recurse(allen_ont)

    return allen2simple


def simplify(
    allen: PathLike | SpatialImage | ArrayLike,
    labels: list[int | str],
    hide_missing: bool = False,
    save: PathLike | bool = False,
) -> SpatialImage | np.ndarray:
    """Simplify an Allen segmentation by collapsing nodes in the ontology."""
    # load/preprocess data
    if isinstance(allen, (str, Path)):
        allen = nb.load(allen)

    if isinstance(allen, SpatialImage):
        allen_dat = allen.dataobj
    else:
        allen_dat = allen

    # prepare linear label maps
    nextbrain2allen = get_allen2simple_map(labels, hide_missing)

    # perform mapping
    allen_dat = nextbrain2allen[np.asarray(allen_dat)]

    if isinstance(allen, SpatialImage):
        header = allen.header
        header.set_data_dtype(allen_dat.dtype)
        allen = type(allen)(allen_dat, None, header)
    else:
        allen = nb.Nifti1Image(allen_dat)

    # save
    if save:
        fname = None
        if isinstance(allen, SpatialImage):
            fname = allen.file_map["image"].filename
        if isinstance(fname, str):
            dirname = op.dirname(fname)
            basename = op.basename(fname)
            basename, ext = op.splitext(basename)
            if ext in (".gz", ".bz2"):
                compression = ext
                basename, ext = op.splitext(basename)
                ext += compression
        else:
            dirname = op.curdir
            basename = "seg"
            ext = ".nii.gz"
        basename += ".simplified"

        if save is True:
            save = f"{dirname}/{basename}{ext}"

        nb.save(allen, save)
        allen = nb.load(save)

    # return
    if isinstance(allen, SpatialImage):
        return allen
    else:
        return allen_dat
