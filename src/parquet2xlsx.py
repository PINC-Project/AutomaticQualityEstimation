import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("excel", type=Path)

    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)
    df.to_excel(args.excel)


if __name__ == "__main__":
    main()
