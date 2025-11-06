"""Command-line interface."""
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

from .combine_hemis import combine_hemis

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
    "combine_hemis",
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
#   Parse and run
# ----------------------------------------------------------------------
args = parser.parse_args()
args.func(args)
