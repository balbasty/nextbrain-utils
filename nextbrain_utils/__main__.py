"""Command-line interface."""
import os.path as op
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

from .combine_hemis import combine_hemis
from .lut import allen_lut
from .simplify import simplify
from .to_allen import to_allen

# ----------------------------------------------------------------------
#   Main parser
# ----------------------------------------------------------------------

parser = ArgumentParser(
    "nextbrain_utils",
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


parser_allen = subparsers.add_parser(
    "allen",
    help="Convert NextBrain labels to Allen Brain labels."
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
#   Write LUTs
# ----------------------------------------------------------------------

LUTDIR = op.join(op.dirname(__file__), "lut")
NEXTBRAIN_LUT = op.join(LUTDIR, "NextBrainLUT.txt")
FREESURFER_LUT = op.join(LUTDIR, "FreeSurferColorLUT.txt")
SYNTHSEG_LUT = op.join(LUTDIR, "SynthSegLUT.txt")


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
    elif args.lut == "synthseg":
        args.output = args.output or "SyntSegLUT.txt"
        shutil.copyfile(SYNTHSEG_LUT, args.output)
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
    choices=("allen", "allen+dk", "nextbrain", "synthseg", "freesurfer"),
    default="allen",
    help=(
        "Lookup table to write: "
        "allen = Labels from the Allen Brain developmental ontology, "
        "allen+dk = Idem and append Desikan-Killiany cortical labels, "
        "nextbrain = Default NextBrain labels, "
        "synthseg = Default SynthSeg labels, "
        "freesurfer = Complete FreeSurfer colormap."
    )
)
parser_lut.add_argument(
    "-a", "--acronym", action="store_true", default=False,
    help="Use Allen Brain acronyms instead of full names."
)
parser_lut.set_defaults(func=_allen_lut)


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
#   Parse and run
# ----------------------------------------------------------------------
args = parser.parse_args()
args.func(args)
