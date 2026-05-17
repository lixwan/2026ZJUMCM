from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from problem2.sim import main


if __name__ == "__main__":
    data_path = PROJECT_ROOT / "data" / "raw" / "teams.csv"
    output_dir = PROJECT_ROOT / "results" / "problem2"
    main(csv_path=data_path, output_dir=output_dir, n_sim=10000)
    print("Problem 2 pipeline finished.")
