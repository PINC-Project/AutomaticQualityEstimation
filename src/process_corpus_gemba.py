import argparse
from pathlib import Path
import pandas as pd

from gemba.utils import get_gemba_scores


def remove_disfluency(t: str) -> str:
    return " ".join(filter(lambda w: "~" not in w and "+" not in w, t.split()))


parser = argparse.ArgumentParser()
parser.add_argument("corpus", type=Path)
parser.add_argument("comet", type=Path)

args = parser.parse_args()

df = pd.read_parquet(str(args.corpus))


source = df["source"].tolist()
hypothesis = df["translation"].tolist()


scores = get_gemba_scores(
    source=source,
    hypothesis=hypothesis,
    source_lang="English",
    target_lang="Polish",
    method="GEMBA-DA",
    model="gpt-4",
)

df["gemba_score"] = scores

df.to_parquet(args.comet)
