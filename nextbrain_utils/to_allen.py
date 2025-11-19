"""Convert NextBrain labels to Allen ontology labels."""
__author__ = "Yael Balbastre, Laura Boettcher"

# std
import os.path as op
import re
from enum import StrEnum
from pathlib import Path

# externals
import nibabel as nb
import numpy as np
from nibabel.spatialimages import SpatialImage
from numpy.typing import ArrayLike

from nextbrain_utils.io import load_lut, load_ontology

PATH_ALLEN = op.join(op.dirname(__file__), "lut", "AllenBrainOntologyDev.json")
PATH_NEXTBRAIN = op.join(op.dirname(__file__), "lut", "NextBrainLUT.txt")

PathLike = str | Path


class CortexOntology(StrEnum):
    """Ontology to use for cortical labels."""

    dev = developmental = "developmental"
    gyr = gyral = "gyral"
    dk = desikankilliany = "desikan-killiany"


def to_allen(
    nextbrain: PathLike | SpatialImage | ArrayLike,
    ontology: CortexOntology | str = CortexOntology.gyral,
    compat16bits: bool = False,
    save: PathLike | bool = False
) -> SpatialImage | np.ndarray:
    """
    Convert NextBrain labels to Allen ontology labels.

    Parameters
    ----------
    nextbrain : PathLike | nb.SpatialImage
        NextBrain segmentation.
    ontology : CortexOntology | str = CortexOntology.gyral
        Ontology to use for cortical labels:
          - developmental: developmental lobules (default)
        - gyral: gyral-based lobules
        - desikan-killiany: Desikan-Killiany atlas
    compat16bits : bool = False
        Whether to convert labels to be compatible with int16.
        (i.e. limit maximum label ID to 32768).
    save:  PathLike | bool = True
        Whether to save the converted segmentation to disk.

    Returns
    -------
    allen: SpatialImage
        Allen segmentation.
    """
    ontology = _ensure_cortex_onto(ontology)

    # load/preprocess data
    if isinstance(nextbrain, (str, Path)):
        nextbrain = nb.load(nextbrain)

    if isinstance(nextbrain, SpatialImage):
        nextbrain_dat = nextbrain.dataobj
    else:
        nextbrain_dat = nextbrain

    # prepare linear label maps
    nextbrain2allen = get_nextbrain2allen_map(ontology)

    if compat16bits:
        # Most Allen labels use 5 digits (e.g. 10962)
        # - Some labels use the prefix 146035, followed by 3 digits.
        #   All suffixes are below between [0, 199].
        # - Some labels use the prefix 146034, followed by 3 digits.
        #   All suffixes are below between [600, 999].
        # - Some labels use the prefix 26644, followed by 4 digits.
        #   All suffixes are below between [1000, 1999].
        # - Some labels use the prefix 26749, followed by 4 digits.
        #   All suffixes are between [9000, 9999].
        # We can safely remove these two prefixes, ensuring that
        # labels remain unique and below 32768.
        #
        # However, the range [1000, 3000] is already used by the
        # DK cortical labels, so we also remap the range prefixed
        # by 26644 to [3000, 4000] instead..
        prefix_146035 = (nextbrain2allen // 1000) == 146035
        prefix_146034 = (nextbrain2allen // 1000) == 146034
        prefix_266441 = (nextbrain2allen // 1000) == 266441
        prefix_267499 = (nextbrain2allen // 1000) == 267499

        nextbrain2allen[prefix_146035] -= 146035000
        nextbrain2allen[prefix_146034] -= 146034000
        nextbrain2allen[prefix_266441] -= (266440000 - 2000)
        nextbrain2allen[prefix_267499] -= 267490000

        nextbrain2allen = nextbrain2allen.astype('i2')

    # perform mapping
    allen_dat = nextbrain2allen[np.asarray(nextbrain_dat)]

    # make SpatialImage
    if isinstance(nextbrain, SpatialImage):
        header = nextbrain.header
        header.set_data_dtype(allen_dat.dtype)
        allen = type(nextbrain)(allen_dat, None, header)
    else:
        allen = nb.Nifti1Image(allen_dat)

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
        basename += ".{ontology}"

        if save is True:
            save = f"{dirname}/{basename}{ext}"
            save = save.format(ontology=str(ontology))

        nb.save(allen, save)
        allen = nb.load(save)

    # return
    if isinstance(nextbrain, SpatialImage):
        return allen
    else:
        return allen_dat


def get_nextbrain2allen_map(
    cortex_ontology: CortexOntology = CortexOntology.gyral
) -> np.ndarray:
    """Compute linear label maps."""
    cortex_ontology = _ensure_cortex_onto(cortex_ontology)

    # load lookup tables
    nextbrain_lut = load_lut()
    allen_ont = load_ontology()
    allen_dtype = 'i4'

    # prepare linear label maps
    max_nextbrain_label = nextbrain_lut["ID"].max()
    nextbrain2allen = np.arange(max_nextbrain_label+1, dtype=allen_dtype)

    # normalize nextbrain names
    nextbrain_norm = {
        elem: normalize_name(elem)
        for elem in nextbrain_lut.names
        if not elem.startswith("ctx-lh-")  # nextbrain always uses RH labels
    }

    # hand-fix cortical names
    if cortex_ontology == CortexOntology.gyral:
        cortex_map = _fscortex_to_allen_gyral()
    elif cortex_ontology == CortexOntology.developmental:
        cortex_map = _fscortex_to_allen_dev()
    else:
        cortex_map = None
    if cortex_map:
        for key in nextbrain_norm.keys():
            if key.startswith("ctx-"):
                nextbrain_norm[key] = cortex_map[key[7:]]

    # Map NextBrain labels to Allen labels
    def _recurse(ont: dict) -> None:
        allen_norm = normalize_name(ont["name"])
        for nxb_orig, nbx_norm in nextbrain_norm.items():
            if allen_norm == nbx_norm:
                nxb_label = nextbrain_lut[
                    nextbrain_lut["NAME"] == nxb_orig.encode()
                ]["ID"]
                for nxb_label1 in nxb_label:
                    nextbrain2allen[nxb_label1] = ont["id"]
        for child in ont.get("children", []):
            _recurse(child)

    _recurse(allen_ont)

    return nextbrain2allen


def _ensure_cortex_onto(x: str | CortexOntology) -> CortexOntology:
    return CortexOntology(getattr(CortexOntology, x, x))


def normalize_name(name: str) -> str:
    """
    Normalize a name by:
      - Lowercasing.
      - Removing commas.
      - Replacing dashes and underscores with spaces.
      - Removing a leading "cortex " if present.
      - Removing any standalone "left" or "right".
      - Collapsing multiple spaces.
    """
    name = name.lower().strip()
    # Remove a leading "cortex " if present.
    if name.startswith("ctx-"):
        name = name[4:]
    # Remove a leading "lh/rh " if present.
    if name.startswith(("lh-", "rh-")):
        name = name[3:]
    # # Remove any standalone "left" or "right" (regardless of position).
    # name = re.sub(r'\b(left|right)\b', '', name)
    # Remove commas.
    name = name.replace(',', '')
    # Replace dashes and underscores with spaces.
    name = name.replace('-', ' ').replace('_', ' ')
    # Collapse multiple spaces.
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def _fscortex_to_allen_guide() -> dict:
    """Convert freesurfer cortical labels to Allen (guide) ontology."""
    # NOTE
    #   * there does not seem to be "cortical corpus callosum" in allen,
    #     but it seems it is also absent from nextbrain.
    #   * entorhinal is part of parahippocampal in allen
    #   * allen does not have a pericalcarine label; it assigns its
    #     superior part to the cuneus and its inferior part to the lingual
    #     gyrus. Since we cannot split our pericalcarine label, we assign
    #     it to the occipital lobe.
    convert = dict()
    convert["unknown"] = "cerebral cortex"
    convert["bankssts"] = "superior temporal gyrus"
    convert["caudalanteriorcingulate"] = "cingulate gyrus frontal part"
    convert["caudalmiddlefrontal"] = "middle frontal gyrus"
    convert["corpuscallosum"] = ""
    convert["cuneus"] = "cuneus"
    convert["entorhinal"] = "parahippocampal gyrus"
    convert["fusiform"] = "fusiform gyrus"
    convert["inferiorparietal"] = "inferior parietal lobule"
    convert["inferiortemporal"] = "inferior temporal gyrus"
    convert["isthmuscingulate"] = "cingulate gyrus retrosplenial part"
    convert["lateraloccipital"] = "occipital lobe"
    convert["lateralorbitofrontal"] = "lateral orbital gyrus"
    convert["lingual"] = "lingual gyrus"
    convert["medialorbitofrontal"] = "gyrus rectus"
    convert["middletemporal"] = "middle temporal gyrus"
    convert["parahippocampal"] = "parahippocampal gyrus"
    convert["paracentral"] = "paracentral lobule anterior part"
    convert["parsopercularis"] = "inferior frontal gyrus opercular part"
    convert["parsorbitalis"] = "inferior frontal gyrus orbital part"
    convert["parstriangularis"] = "inferior frontal gyrus triangular part"
    convert["pericalcarine"] = "occipital lobe"
    convert["postcentral"] = "postcentral gyrus"
    convert["posteriorcingulate"] = "cingulate gyrus parietal part"
    convert["precentral"] = "precentral gyrus"
    convert["precuneus"] = "precuneus"
    convert["rostralanteriorcingulate"] = "cingulate gyrus frontal part"
    convert["rostralmiddlefrontal"] = "middle frontal gyrus"
    convert["superiorfrontal"] = "superior frontal gyrus"
    convert["superiorparietal"] = "superior parietal lobule"
    convert["superiortemporal"] = "superior temporal gyrus"
    convert["supramarginal"] = "supramarginal gyrus"
    convert["frontalpole"] = "frontal pole"
    convert["temporalpole"] = "temporal pole"
    convert["transversetemporal"] = "transverse gyri"
    convert["insula"] = "insula"
    return convert


def _fscortex_to_allen_gyral() -> dict:
    """
    Convert freesurfer cortical labels to Allen (developmental lobules)
    ontology.
    """
    # NOTE
    #   * there does not seem to be "cortical corpus callosum" in allen,
    #     but it seems it is also absent from nextbrain.
    #   * FS labels the banks of the superior temporal sulcus with the
    #     same label (banksts), whereas allen assigns the superior bank
    #     to the parietal lobe and the inferior bank to the temporal lobe.
    #     We therefore assign the label to "cerebral gyri and lobules"
    #   * FS separates the caudal and rostral parts of the middle frontal gyrus
    #     but allen does not. They threfore get merged together.
    #   * FS's entorhinal cortex roughly corresponds to Allen's anterior
    #     parahipocampal gyrus, and FS's parahipocampal roughly corresponds
    #     to Allen's posrterior parahipocampal gyrus.
    #   * FS's lateral occipital cortex is split across Allen's suiperior
    #     and inferior occipital. We therefore assign it to "occipital lobe".
    #   * FS's paracentral cortex is split across an anterior part
    #     (frontal lobe) and posterior part (parietal lobe). I choose
    #     to assign it to the parietal lobe.
    #   * Allen does not have a pericalcarine label; it assigns its
    #     superior part to the cuneus and its inferior part to the lingual
    #     gyrus. Since we cannot split our pericalcarine label, we assign
    #     it to the occipital lobe.
    convert = dict()
    convert["unknown"] = "cerebral gyri and lobules"
    convert["bankssts"] = "cerebral gyri and lobules"
    convert["caudalanteriorcingulate"] = "cingulate gyrus caudal (posterior) part"  # noqa: E501
    convert["caudalmiddlefrontal"] = "middle frontal gyrus"
    convert["corpuscallosum"] = ""
    convert["cuneus"] = "cuneus"
    convert["entorhinal"] = "anterior parahippocampal gyrus"
    convert["fusiform"] = "occipitotemporal (fusiform) gyrus occipital part"
    convert["inferiorparietal"] = "inferior parietal lobule"
    convert["inferiortemporal"] = "inferior temporal gyrus"
    convert["isthmuscingulate"] = "cingulate gyrus caudal (posterior) part"
    convert["lateraloccipital"] = "occipital lobe"
    convert["lateralorbitofrontal"] = "lateral orbital gyrus"
    convert["lingual"] = "lingual gyrus"
    convert["medialorbitofrontal"] = "gyrus rectus (straight gyrus)"
    convert["middletemporal"] = "middle temporal gyrus"
    convert["parahippocampal"] = "posterior parahippocampal gyrus"
    convert["paracentral"] = "paracentral lobule rostral part"
    convert["parsopercularis"] = "inferior frontal gyrus opercular part"
    convert["parsorbitalis"] = "inferior frontal gyrus orbital part"
    convert["parstriangularis"] = "inferior frontal gyrus triangular part"
    convert["pericalcarine"] = "occipital lobe"
    convert["postcentral"] = "postcentral gyrus"
    convert["posteriorcingulate"] = "cingulate gyrus caudal (posterior) part"
    convert["precentral"] = "precentral gyrus"
    convert["precuneus"] = "precuneus"
    convert["rostralanteriorcingulate"] = "cingulate gyrus rostral (anterior) part"  # noqa: E501
    convert["rostralmiddlefrontal"] = "middle frontal gyrus"
    convert["superiorfrontal"] = "superior frontal gyrus"
    convert["superiorparietal"] = "supraparietal lobule"
    convert["superiortemporal"] = "superior temporal gyrus"
    convert["supramarginal"] = "supramarginal gyrus"
    convert["frontalpole"] = "frontal pole"
    convert["temporalpole"] = "temporal pole"
    convert["transversetemporal"] = "transverse temporal gyrus (Heschl's gyrus)"  # noqa: E501
    convert["insula"] = "insular lobe"
    return convert


def _fscortex_to_allen_dev() -> dict:
    """
    Convert freesurfer cortical labels to Allen (developmental brodmann)
    ontology.
    """
    convert = dict()
    convert["unknown"] = "neocortex (isocortex)"
    convert["bankssts"] = "temporal neocortex"
    convert["caudalanteriorcingulate"] = "cingulate neocortex"
    convert["caudalmiddlefrontal"] = "frontal neocortex"
    convert["corpuscallosum"] = ""
    convert["cuneus"] = "occipital neocortex"
    convert["entorhinal"] = "periarchicortex"
    convert["fusiform"] = "temporal neocortex"
    convert["inferiorparietal"] = "parietal neocortex"
    convert["inferiortemporal"] = "temporal neocortex"
    convert["isthmuscingulate"] = "cingulate neocortex"
    convert["lateraloccipital"] = "occipital neocortex"
    convert["lateralorbitofrontal"] = "frontal neocortex"
    convert["lingual"] = "occipital neocortex"
    convert["medialorbitofrontal"] = "frontal neocortex"
    convert["middletemporal"] = "temporal neocortex"
    convert["parahippocampal"] = "periarchicortex"
    convert["paracentral"] = "frontal neocortex"
    convert["parsopercularis"] = "frontal neocortex"
    convert["parsorbitalis"] = "frontal neocortex"
    convert["parstriangularis"] = "frontal neocortex"
    convert["pericalcarine"] = "occipital neocortex"
    convert["postcentral"] = "parietal neocortex"
    convert["posteriorcingulate"] = "cingulate neocortex"
    convert["precentral"] = "frontal neocortex"
    convert["precuneus"] = "parietal neocortex"
    convert["rostralanteriorcingulate"] = "cingulate neocortex"
    convert["rostralmiddlefrontal"] = "frontal neocortex"
    convert["superiorfrontal"] = "frontal neocortex"
    convert["superiorparietal"] = "parietal neocortex"
    convert["superiortemporal"] = "temporal neocortex"
    convert["supramarginal"] = "parietal neocortex"
    convert["frontalpole"] = "frontal neocortex"
    convert["temporalpole"] = "temporal neocortex"
    convert["transversetemporal"] = "temporal neocortex"
    convert["insula"] = "insular neocortex"
    return convert
