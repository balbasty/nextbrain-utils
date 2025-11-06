"""Combine left and right sides into a single file."""
import os.path as op
from itertools import product
from pathlib import Path

import nibabel as nb
import numpy as np
from nibabel.spatialimages import SpatialImage

PathLike = str | Path


def combine_hemis(
    left: PathLike | SpatialImage,
    right: PathLike | SpatialImage,
    save: bool | PathLike = True,
    save_sides: bool | PathLike = True,
) -> SpatialImage:
    """
    Combine NextBrain hemispheres into a single file.

    Parameters
    ----------
    left : PathLike | nb.SpatialImage
        NextBrain segmentation of the left hemisphere.
    right : PathLike | nb.SpatialImage
        NextBrain segmentation of the right hemisphere.
    save:  PathLike | bool = True
        Whether to save the combined segmentation to disk.
    save_side: PathLike | bool = True
        Whether to save the lateralization mask to disk.

    Returns
    -------
    combined : nb.SpatialImage
        The combined NextBrain segmentation.
    sides : nb.SpatialImage
        A mask of the left (label = 1) and right (label = 2) hemispheres.

    """
    if isinstance(left, (str, Path)):
        left = nb.load(left)
    if isinstance(right, (str, Path)):
        right = nb.load(right)

    bbox = list(product([0, 1], [0, 1], [0, 1]))
    right2left = np.linalg.inv(left.affine) @ right.affine

    # coordinate of the corners of the left image, in left voxel space
    left_shape = np.asarray(left.shape[:3])
    left_bbox = np.asarray(bbox)
    left_bbox = left_bbox * (left_shape - 1)[None, :]
    left_bbox_min = np.round(left_bbox.min(0)).astype("i8")
    left_bbox_max = np.round(left_bbox.max(0)).astype("i8")

    # coordinate of the corners of the right image, in right voxel space
    right_shape = np.asarray(right.shape[:3])
    right_bbox = np.asarray(bbox)
    right_bbox = right_bbox * (right_shape - 1)[None, :]

    # convert right corners to left voxel space
    right_bbox = right_bbox @ right2left[:3, :3].T + right2left[:3, -1]
    right_bbox_min = np.round(right_bbox.min(0)).astype("i8")
    right_bbox_max = np.round(right_bbox.max(0)).astype("i8")

    # compute the shape of the combined volume
    bmin = np.minimum(left_bbox_min, right_bbox_min)
    bmax = np.maximum(left_bbox_max, right_bbox_max)
    shape = bmax - bmin + 1

    # compute the bounding box of the left and right images in combined space
    left_bbox = tuple(
        slice(
            int(np.round(left_bbox_min[i]-bmin[i]).item()),
            int(np.round(left_bbox_max[i]-bmin[i]).item()) + 1
        )
        for i in range(3)
    )
    right_bbox = tuple(
        slice(
            int(np.round(right_bbox_min[i]-bmin[i]).item()),
            int(np.round(right_bbox_max[i]-bmin[i]).item()) + 1
        )
        for i in range(3)
    )

    # compute the affine of the combined volume
    comb2left = np.eye(4)
    comb2left[:3, -1] = bmin - left_bbox_min
    affine = left.affine @ comb2left

    comb2right = np.eye(4)
    comb2right[:3, -1] = bmin - right_bbox_min

    # combine
    out = np.zeros(shape, dtype=left.dataobj.dtype)
    out[left_bbox] += left.dataobj
    out[right_bbox] += right.dataobj

    # create nibabel spatial object
    out = type(left)(out, affine, left.header)

    # same for left/right mask
    msk = np.zeros(shape, dtype="u1")
    msk[left_bbox] += (np.asarray(left.dataobj) > 0) * np.uint8(1)
    msk[right_bbox] += (np.asarray(right.dataobj) > 0) * np.uint8(2)

    msk_header = left.header
    msk_header.set_data_dtype("u1")
    msk = type(left)(msk, affine, msk_header)

    if save or save_sides:
        fname = left.file_map["image"].filename
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
        if ".left" in basename:
            basename = basename.replace(".left", ".{side}")
        else:
            basename += ".{side}"

    if save:
        if save is True:
            save = f"{dirname}/{basename}{ext}"
            save = save.format(side="combined")
        nb.save(out, save)
        out = nb.load(save)

    if save_sides:
        if save_sides is True:
            save_sides = f"{dirname}/{basename}{ext}"
            save_sides = save_sides.format(side="sides")
        nb.save(msk, save_sides)
        msk = nb.load(save_sides)

    return out, msk
