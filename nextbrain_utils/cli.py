"""Command-line interface."""
import os.path as op
import shutil
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
    RawDescriptionHelpFormatter,
)

from .combine_hemis import combine_hemis
from .lut import allen_lut
from .simplify import simplify
from .to_allen import to_allen
from .to_aseg import to_aseg
from .to_supersynth import to_supersynth

# ----------------------------------------------------------------------
#   Main parser
# ----------------------------------------------------------------------

parser = ArgumentParser(
    "nextbrain-utils",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
subparsers = parser.add_subparsers(required=True)


# ----------------------------------------------------------------------
#   Combine Hemisphere
# ----------------------------------------------------------------------

def _combine_hemis(args: Namespace) -> None:
    if len(args.left) != len(args.right):
        raise ValueError("Number of left and right files do not match.")
    if len(args.output) == 0:
        args.output = [True] * len(args.left)
    elif len(args.output) == 1:
        args.output *= len(args.left)
    elif len(args.output) != len(args.left):
        raise ValueError("Number of left and output files do not match.")
    if len(args.output_sides) == 0:
        args.output_sides = [True] * len(args.left)
    elif len(args.output_sides) == 1:
        args.output_sides *= len(args.left)
    elif len(args.output_sides) != len(args.left):
        raise ValueError("Number of left and mask files do not match.")

    for left, right, out, sides in zip(
        args.left, args.right, args.output, args.output_sides
    ):
        combine_hemis(left, right, out, sides)


parser_combine = subparsers.add_parser(
    "combine",
    help="Combine two NextBrain hemispheres into a single file."
)
parser_combine.add_argument(
    "-l", "--left", nargs="+", help="Left hemisphere(s)."
)
parser_combine.add_argument(
    "-r", "--right", nargs="+", help="Right hemisphere(s)."
)
parser_combine.add_argument(
    "-o", "--output", nargs="+", default=[],
    help="Output filename(s) of the combined segmentation."
)
parser_combine.add_argument(
    "-m", "--output-sides", nargs="+", default=[],
    help="Output filename(s) of the laterlization mask."
)
parser_combine.set_defaults(func=_combine_hemis)


# ----------------------------------------------------------------------
#   Convert to Allen
# ----------------------------------------------------------------------

def _to_allen(args: Namespace) -> None:
    if len(args.output) == 0:
        args.output = [True] * len(args.input)
    elif len(args.output) == 1:
        args.output *= len(args.input)
    elif len(args.output) != len(args.input):
        raise ValueError("Number of input and output files do not match.")

    for inp, out in zip(args.input, args.output):
        to_allen(inp, args.cortex_ontology, out)


_desc = """Convert NextBrain labels to Allen Brain labels.
===============================================

NextBrain does not use the Allen ontology for its cortical labels, but
instead uses the Desikan-Killiany labels (i.e., Freesurfer's "aparc").

We implement three different ways to deal with them. They can be
selected using the `--cortex-ontology` option.

1) dk = Freesurfer's Desikan-Killiany labels
--------------------------------------------

This option preserves the Desikan-Killiany labels and their IDs (which
happen to not be used by any region in the Allen ontology). In a
Freesurfer "aseg+aparc" file, different label values are used in the left
and right hemisphere, whereas nextbrain uses the right hemisphere label
values in both hemispheres. In the "allen+dk" lookup table (see `lut -h`)
the hemisphere ("lh" or "rh") is stripped from the region name.

2000  ctx-unknown
2001  ctx-bankssts
2002  ctx-caudalanteriorcingulate
2003  ctx-caudalmiddlefrontal
2004  ctx-corpuscallosum
2005  ctx-cuneus
2006  ctx-entorhinal
2007  ctx-fusiform
2008  ctx-inferiorparietal
2009  ctx-inferiortemporal
2010  ctx-isthmuscingulate
2011  ctx-lateraloccipital
2012  ctx-lateralorbitofrontal
2013  ctx-lingual
2014  ctx-medialorbitofrontal
2015  ctx-middletemporal
2016  ctx-parahippocampal
2017  ctx-paracentral
2018  ctx-parsopercularis
2019  ctx-parsorbitalis
2020  ctx-parstriangularis
2021  ctx-pericalcarine
2022  ctx-postcentral
2023  ctx-posteriorcingulate
2024  ctx-precentral
2025  ctx-precuneus
2026  ctx-rostralanteriorcingulate
2027  ctx-rostralmiddlefrontal
2028  ctx-superiorfrontal
2029  ctx-superiorparietal
2030  ctx-superiortemporal
2031  ctx-supramarginal
2032  ctx-frontalpole
2033  ctx-temporalpole
2034  ctx-transversetemporal
2035  ctx-insula

2) gyr = Allen Brain's gyral ontology
-------------------------------------

Freesurfer's Desikan-Killiany labels are mostly gyral-based, and the
Allen Brain ontology also defines gyral cortical labels. However, there
is no exact one-to-one mapping between the two schemes. We therefore
use a "best guess" mapping.

2000  ctx-unknown                   ->     12112  cerebral gyri and lobules
2001  ctx-bankssts                  ->     12112  cerebral gyri and lobules
2002  ctx-caudalanteriorcingulate   ->     12158  cingulate gyrus, caudal (posterior) part
2003  ctx-caudalmiddlefrontal       ->     12116  middle frontal gyrus
2004  ctx-corpuscallosum            -> xxxxxxxxx  not in nextbrain or allen
2005  ctx-cuneus                    ->     12150  cuneus
2006  ctx-entorhinal                ->     12163  anterior parahippocampal gyrus
2007  ctx-fusiform                  ->     12152  occipitotemporal (fusiform) gyrus, occipital part
2008  ctx-inferiorparietal          ->     12134  inferior parietal lobule
2009  ctx-inferiortemporal          ->     12142  inferior temporal gyrus
2010  ctx-isthmuscingulate          ->     12158  cingulate gyrus, caudal (posterior) part
2011  ctx-lateraloccipital          ->     12148  occipital lobe
2012  ctx-lateralorbitofrontal      ->     12125  lateral orbital gyrus
2013  ctx-lingual                   ->     12151  lingual gyrus
2014  ctx-medialorbitofrontal       ->     12121  gyrus rectus (straight gyrus)
2015  ctx-middletemporal            ->     12141  middle temporal gyrus
2016  ctx-parahippocampal           ->     12164  posterior parahippocampal gyrus
2017  ctx-paracentral               ->     12138  paracentral lobule, rostral part
2018  ctx-parsopercularis           ->     12119  inferior frontal gyrus, opercular part
2019  ctx-parsorbitalis             ->     12120  inferior frontal gyrus, orbital part
2020  ctx-parstriangularis          ->     12118  inferior frontal gyrus, triangular part
2021  ctx-pericalcarine             ->     12148  occipital lobe
2022  ctx-postcentral               ->     12132  postcentral gyrus
2023  ctx-posteriorcingulate        ->     12158  cingulate gyrus, caudal (posterior) part
2024  ctx-precentral                ->     12114  precentral gyrus
2025  ctx-precuneus                 ->     12137  precuneus
2026  ctx-rostralanteriorcingulate  ->     12157  cingulate gyrus, rostral (anterior) part
2027  ctx-rostralmiddlefrontal      ->     12116  middle frontal gyrus
2028  ctx-superiorfrontal           ->     12115  superior frontal gyrus
2029  ctx-superiorparietal          ->     12133  supraparietal lobule
2030  ctx-superiortemporal          ->     12140  superior temporal gyrus
2031  ctx-supramarginal             ->     12135  supramarginal gyrus
2032  ctx-frontalpole               -> 146034888  frontal pole
2033  ctx-temporalpole              ->     12146  temporal pole
2034  ctx-transversetemporal        ->     12144  transverse temporal gyrus (Heschl's gyrus)
2035  ctx-insula                    ->     12176  insular lobe

Notes:
* DK labels both banks of the superior temporal sulcus with the
  same label ("banksts"), whereas Allen assigns the superior bank
  to the parietal lobe and the inferior bank to the temporal lobe.
  We therefore assign the label to "cerebral gyri and lobules".
* DK separates the caudal and rostral parts of the middle frontal gyrus
  but allen does not. They therefore get merged together.
* DK's entorhinal cortex roughly corresponds to Allen's anterior
  parahipocampal gyrus, and FS's parahipocampal roughly corresponds
  to Allen's posterior parahipocampal gyrus.
* DK's lateral occipital cortex is split across Allen's superior
  and inferior occipital. We therefore assign it to "occipital lobe".
* DK's paracentral cortex is split across an anterior part
  (frontal lobe) and posterior part (parietal lobe). Following the
  Freesurfer documentation, we choose to assign it to the fontal lobe.
* Allen does not have a pericalcarine label; it assigns its
  superior part to the cuneus and its inferior part to the lingual
  gyrus. Since we cannot split our pericalcarine label, we assign
  it to the occipital lobe.

3) dev = Allen Brain's developmental ontology
---------------------------------------------

Allen's gyral hierarchy is part of their "developmental" ontology but
forms a separate hierarchy under "cerebral gyri and lobules". The "dev"
option use cortical labels from Allen's main hierarchy, whose cortical
leaves are Brodmann areas (BAs). Since we cannot map DK labels to BAs,
we combine them into the main subdivisions of the neocortex, (mostly)
following the Freesurfer documentation.

10160   neocortex (isocortex)   <- 2000  ctx-unknown

10161   frontal neocortex       <- 2003  ctx-caudalmiddlefrontal
                                <- 2012  ctx-lateralorbitofrontal
                                <- 2014  ctx-medialorbitofrontal
                                <- 2017  ctx-paracentral
                                <- 2018  ctx-parsopercularis
                                <- 2019  ctx-parsorbitalis
                                <- 2020  ctx-parstriangularis
                                <- 2024  ctx-precentral
                                <- 2027  ctx-rostralmiddlefrontal
                                <- 2028  ctx-superiorfrontal
                                <- 2032  ctx-frontalpole

10208   parietal neocortex      <- 2008  ctx-inferiorparietal
                                <- 2022  ctx-postcentral
                                <- 2025  ctx-precuneus
                                <- 2029  ctx-superiorparietal
                                <- 2031  ctx-supramarginal

10235   temporal neocortex      <- 2001  ctx-bankssts
                                <- 2007  ctx-fusiform
                                <- 2009  ctx-inferiortemporal
                                <- 2015  ctx-middletemporal
                                <- 2030  ctx-superiortemporal
                                <- 2033  ctx-temporalpole
                                <- 2034  ctx-transversetemporal

10268   occipital neocortex     <- 2005  ctx-cuneus
                                <- 2011  ctx-lateraloccipital
                                <- 2013  ctx-lingual
                                <- 2021  ctx-pericalcarine

10288   insular neocortex       <- 2035  ctx-insula

10314   periarchicortex         <- 2006  ctx-entorhinal
                                <- 2016  ctx-parahippocampal

"""  # noqa: E501


parser_allen = subparsers.add_parser(
    "allen",
    description=_desc,
    help="Convert NextBrain labels to Allen Brain labels.",
    formatter_class=RawDescriptionHelpFormatter,
)
parser_allen.add_argument(
    "-i", "--input", nargs="+", help="Input segmentation(s)."
)
parser_allen.add_argument(
    "-o", "--output", nargs="+", default=[],
    help="Output filename(s) of the segmentation(s)."
)
parser_allen.add_argument(
    "-c", "--cortex-ontology", choices=("gyr", "dev", "dk"), default="gyr",
    help=(
        "Ontology to use when converting cortical labels: "
        "gyr = Allen's gyral ontology, "
        "dev = Allen's developmental ontology, "
        "dk = Freesurfer's Desikan-Killiany labels."
    )
)
parser_allen.set_defaults(func=_to_allen)


# ----------------------------------------------------------------------
#   Convert to ASeg+AParc
# ----------------------------------------------------------------------

def _to_aseg(args: Namespace) -> None:
    if len(args.output) == 0:
        args.output = [True] * len(args.input)
    elif len(args.output) == 1:
        args.output *= len(args.input)
    elif len(args.output) != len(args.input):
        raise ValueError("Number of input and output files do not match.")

    if len(args.side) == 0:
        args.side = ["R"] * len(args.input)
    elif len(args.side) == 1:
        args.side *= len(args.input)
    elif len(args.side) != len(args.input):
        raise ValueError("Number of input files and sides do not match.")

    for inp, side, out in zip(args.input, args.side, args.output):
        to_aseg(inp, side, args.claustrum, out)


parser_aseg = subparsers.add_parser(
    "aseg",
    help="Convert NextBrain labels to ASeg+AParc labels.",
)
parser_aseg.add_argument(
    "-i", "--input", nargs="+", help="Input segmentation(s)."
)
parser_aseg.add_argument(
    "-o", "--output", nargs="+", default=[],
    help="Output filename(s) of the segmentation(s)."
)
parser_aseg.add_argument(
    "-s", "--side", nargs="+", default=["right"],
    help="Side of the hemisphere (left or right), or path to side label map."
)
parser_aseg.add_argument(
    "-c", "--claustrum", action="store_true", default=False,
    help=(
        "Include a claustrum label. "
        "Otherwise, claustrum voxels are assigned to white matter."
    )
)
parser_aseg.set_defaults(func=_to_aseg)


# ----------------------------------------------------------------------
#   Convert to SuperSynth
# ----------------------------------------------------------------------

def _to_supersynth(args: Namespace) -> None:
    if len(args.output) == 0:
        args.output = [True] * len(args.input)
    elif len(args.output) == 1:
        args.output *= len(args.input)
    elif len(args.output) != len(args.input):
        raise ValueError("Number of input and output files do not match.")

    if len(args.side) == 0:
        args.side = ["R"] * len(args.input)
    elif len(args.side) == 1:
        args.side *= len(args.input)
    elif len(args.side) != len(args.input):
        raise ValueError("Number of input files and sides do not match.")

    for inp, side, out in zip(args.input, args.side, args.output):
        to_supersynth(inp, side, args.claustrum, out)


parser_supersynth = subparsers.add_parser(
    "supersynth",
    help="Convert NextBrain labels to SuperSynth labels.",
)
parser_supersynth.add_argument(
    "-i", "--input", nargs="+", help="Input segmentation(s)."
)
parser_supersynth.add_argument(
    "-o", "--output", nargs="+", default=[],
    help="Output filename(s) of the segmentation(s)."
)
parser_supersynth.add_argument(
    "-s", "--side", nargs="+", default=["right"],
    help="Side of the hemisphere (left or right), or path to side label map."
)
parser_supersynth.add_argument(
    "-c", "--claustrum", action="store_true", default=False,
    help=(
        "Include a claustrum label. "
        "Otherwise, claustrum voxels are assigned to white matter."
    )
)
parser_supersynth.set_defaults(func=_to_supersynth)


# ----------------------------------------------------------------------
#   Simplify label map
# ----------------------------------------------------------------------

def _simplify(args: Namespace) -> None:
    if len(args.output) == 0:
        args.output = [True] * len(args.input)
    elif len(args.output) == 1:
        args.output *= len(args.input)
    elif len(args.output) != len(args.input):
        raise ValueError("Number of input and output files do not match.")

    for inp, out in zip(args.input, args.output):
        simplify(inp, args.labels, args.delete_missing, out)


parser_simplify = subparsers.add_parser(
    "simplify",
    help="Simplify Allen Brain label maps."
)
parser_simplify.add_argument(
    "-i", "--input", nargs="+", help="Input segmentation(s)."
)
parser_simplify.add_argument(
    "-o", "--output", nargs="+", default=[],
    help="Output filename(s) of the segmentation(s)."
)
parser_simplify.add_argument(
    "-l", "--labels", nargs="+",
    help=(
        "Name or ID of regions to simplify. "
        "All subregions of the listed regions will be mapped to their parent."
    )
)
parser_simplify.add_argument(
    "-d", "--delete-missing", action="store_true", default=False,
    help="Delete regions that are not listed in --labels"
)
parser_simplify.set_defaults(func=_simplify)


# ----------------------------------------------------------------------
#   Write LUTs
# ----------------------------------------------------------------------

LUTDIR = op.join(op.dirname(__file__), "lut")
NEXTBRAIN_LUT = op.join(LUTDIR, "NextBrainLUT.txt")
FREESURFER_LUT = op.join(LUTDIR, "FreeSurferColorLUT.txt")
ASEG_LUT = op.join(LUTDIR, "ASegLUT.txt")
SUPERSYNTH_LUT = op.join(LUTDIR, "SuperSynthWholeLUT.txt")
SUPERSYNTH_CEREBRUM_LUT = op.join(LUTDIR, "SuperSynthCerebrumLUT.txt")
SUPERSYNTH_EXVIVO_LUT = op.join(LUTDIR, "SuperSynthExVivoLUT.txt")


def _allen_lut(args: Namespace) -> None:
    if "allen" in args.lut:
        args.output = args.output or "AllenBrainLUT.txt"
        allen_lut(
            acronym=args.acronym,
            append_dk="dk" in args.lut,
            save=args.output
        )
    elif args.lut == "nextbrain":
        args.output = args.output or "NextBrainLUT.txt"
        shutil.copyfile(NEXTBRAIN_LUT, args.output)
    elif args.lut == "freesurfer":
        args.output = args.output or "FreeSurferColorLUT.txt"
        shutil.copyfile(FREESURFER_LUT, args.output)
    elif args.lut in ("aseg", "synthseg"):
        args.output = args.output or "ASegLUT.txt"
        shutil.copyfile(ASEG_LUT, args.output)
    elif args.lut == "supersynth":
        args.output = args.output or "SuperSynthWholeLUT.txt"
        shutil.copyfile(SUPERSYNTH_LUT, args.output)
    elif args.lut == "supersynth-cerebrum":
        args.output = args.output or "SuperSynthCerebrumLUT.txt"
        shutil.copyfile(SUPERSYNTH_CEREBRUM_LUT, args.output)
    elif args.lut == "supersynth-exvivo":
        args.output = args.output or "SuperSynthExVivoLUT.txt"
        shutil.copyfile(SUPERSYNTH_EXVIVO_LUT, args.output)
    else:
        raise ValueError(args.lut)


parser_lut = subparsers.add_parser(
    "lut",
    help="Write freesurfer lookup tables."
)
parser_lut.add_argument(
    "-o", "--output", default=None,
    help="Output filename of the lut."
)
parser_lut.add_argument(
    "-l", "--lut",
    choices=(
        "allen", "allen+dk", "nextbrain", "aseg", "freesurfer",
        "supersynth", "supersynth-cerebrum", "supersynth-exvivo",
    ),
    default="allen+dk",
    help=(
        "Lookup table to write: "
        "allen = Labels from the Allen Brain developmental ontology, "
        "allen+dk = Allen + append Desikan-Killiany cortical labels, "
        "nextbrain = NextBrain, "
        "aseg = ASeg+AParc/SynthSeg, "
        "supersynth = SuperSynth."
        "supersynth-cerebrum = SuperSynth (cerebrum mode)."
        "supersynth-exvivo = SuperSynth (exvivo mode)."
        "freesurfer = Complete FreeSurfer colormap, "
    )
)
parser_lut.add_argument(
    "-a", "--acronym", action="store_true", default=False,
    help="Use Allen Brain acronyms instead of full names."
)
parser_lut.set_defaults(func=_allen_lut)


# ----------------------------------------------------------------------
#   Parse and run
# ----------------------------------------------------------------------
def main() -> None:
    """Run the command-line interface."""
    args = parser.parse_args()
    args.func(args)
