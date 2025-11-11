"""Convert NextBrain labels to SuperSynth labels."""
__author__ = "Yael Balbastre"

# std
import os.path as op
from enum import StrEnum
from pathlib import Path

# externals
import nibabel as nb
import numpy as np
from nibabel.spatialimages import SpatialImage
from numpy.typing import ArrayLike

# internals
from .io import load_lut, load_yaml

LUTDIR = op.join(op.dirname(__file__), "lut")
PATH_NEXTBRAIN = op.join(LUTDIR, "NextBrainLUT.txt")
PATH_SUPERSYNTH = op.join(LUTDIR, "SuperSynthWholeLUT.txt")
PATH_MAPPING = op.join(LUTDIR, "NextBrainToSuperSynth.yaml")

PathLike = str | Path


class Side(StrEnum):
    """Brain hemisphere side."""

    LEFT = L = "L"
    RIGHT = R = "R"


def to_supersynth(
    nextbrain: PathLike | SpatialImage | ArrayLike,
    side: str | Side | PathLike | SpatialImage | ArrayLike = Side.L,
    claustrum: bool = False,
    save: PathLike | bool = False
) -> SpatialImage | np.ndarray:
    """
    Convert NextBrain labels to ASeg+AParc labels.

    Parameters
    ----------
    nextbrain : PathLike | nb.SpatialImage
        NextBrain segmentation.
    side : str | Side | PathLike | SpatialImage | array_like = "L"
        Hemisphere side or side segmentation.
    save :  PathLike | bool = True
        Whether to save the converted segmentation to disk.

    Returns
    -------
    allen: SpatialImage
        Allen segmentation.
    """
    # load/preprocess data
    if isinstance(nextbrain, (str, Path)):
        nextbrain = nb.load(nextbrain)
    if isinstance(nextbrain, SpatialImage):
        nextbrain_dat = nextbrain.dataobj
    else:
        nextbrain_dat = nextbrain
    nextbrain_dat = np.asarray(nextbrain_dat)

    if _is_side(side):
        side = _ensure_side(side)
    elif isinstance(side, (str, Path)):
        side = nb.load(side)
    if isinstance(side, SpatialImage):
        side = np.asarray(side.dataobj)

    # prepare linear label maps
    mapping_left = get_nextbrain2supersynth_map("L", claustrum)
    mapping_right = get_nextbrain2supersynth_map("R", claustrum)

    # perform mapping
    if isinstance(side, Side):
        mapping = mapping_left if side == Side.LEFT else mapping_right
        aseg_dat = mapping[nextbrain_dat]
    else:
        aseg_dat = np.zeros_like(nextbrain_dat, dtype=mapping_left.dtype)
        mask = side == 1
        aseg_dat[mask] = mapping_left[nextbrain_dat[mask]]
        mask = side == 2
        aseg_dat[mask] = mapping_right[nextbrain_dat[mask]]

    if isinstance(nextbrain, SpatialImage):
        header = nextbrain.header
        header.set_data_dtype(aseg_dat.dtype)
        aseg = type(nextbrain)(aseg_dat, None, header)
    else:
        aseg = nb.Nifti1Image(aseg_dat)

    # save
    if save:
        fname = None
        if isinstance(nextbrain, SpatialImage):
            fname = nextbrain.file_map["image"].filename
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
        basename += ".aseg+aparc"

        if save is True:
            save = f"{dirname}/{basename}{ext}"

        nb.save(aseg, save)
        aseg = nb.load(save)

    # return
    if isinstance(nextbrain, SpatialImage):
        return aseg
    else:
        return aseg_dat


def get_nextbrain2supersynth_map(
    side: str | Side = "L",
    claustrum: bool = False,
) -> np.ndarray:
    """Compute linear label maps."""
    side = _ensure_side(side)

    # load lookup tables
    nextbrain_lut = load_lut(PATH_NEXTBRAIN)
    supersynth_lut = load_lut(PATH_SUPERSYNTH)
    supersynth_map = load_yaml(PATH_MAPPING)
    dtype = 'i2'

    # prepare linear label maps
    max_nextbrain_label = nextbrain_lut["ID"].max()
    nextbrain2supersynth = np.arange(max_nextbrain_label+1, dtype=dtype)

    lateralized_labels = [
        "Cerebral-White-Matter",
        "Cerebral-Cortex",
        "Lateral-Ventricle",
        "Inf-Lat-Vent",
        "Cerebellum-White-Matter",
        "Cerebellum-Cortex",
        "Thalamus",
        "Caudate",
        "Putamen",
        "Pallidum",
        "Hippocampus",
        "Amygdala",
        "Accumbens-area",
        "VentralDC",
        "Claustrum",
        "HypoThal-noMB",
        "Fornix",
        "MammillaryBody",
        "Basal-Forebrain",
        "SeptalNuc",
    ]

    if side == Side.LEFT:
        # Cortical labels: map right to left
        nextbrain2supersynth[nextbrain2supersynth >= 2000] -= 1000

    side_prefix = "Left-" if side == Side.LEFT else "Right-"
    for supersynth_label, nextbrain_labels in supersynth_map.items():

        if supersynth_label == "Claustrum" and not claustrum:
            supersynth_label = "Cerebral-White-Matter"

        if supersynth_label in lateralized_labels:
            supersynth_label = side_prefix + supersynth_label
        supersynth_id = supersynth_lut.name2label(supersynth_label)

        for label in nextbrain_labels:
            nextbrain2supersynth[label] = supersynth_id

    return nextbrain2supersynth


def _is_side(side: str | Side) -> bool:
    return side.upper() in ("L", "R", "LEFT", "RIGHT")


def _ensure_side(x: str | Side) -> Side:
    return Side(getattr(Side, x.upper(), x))
