import sys
from pathlib import Path

# Add parent directory to path for imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.parser import build_parser
from train.runner import main


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.num_images == 4 and args.num_coefficients != 25:
        print("Warning: 4 images typically used with 25 coefficients")
    if args.num_images == 5 and args.num_coefficients != 77:
        print("Warning: 5 images typically used with 77 coefficients")

    main(args)
