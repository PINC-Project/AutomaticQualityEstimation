import argparse
from pathlib import Path
import pandas as pd
from comet import download_model, load_from_checkpoint


def remove_disfluency(t: str) -> str:
    return " ".join(filter(lambda w: "~" not in w and "+" not in w, t.split()))


parser = argparse.ArgumentParser()
parser.add_argument("corpus", type=Path)
parser.add_argument("comet", type=Path)

args = parser.parse_args()

df = pd.read_parquet(str(args.corpus))


model_path = download_model("Unbabel/wmt20-comet-qe-da")
model = load_from_checkpoint(model_path)

data = [
    {"src": remove_disfluency(row.source), "mt": remove_disfluency(row.translation)}
    for _, row in df.iterrows()
]

model_output = model.predict(data, batch_size=4, gpus=1)

df["comet_score"] = model_output["scores"]

df.to_parquet(args.comet)
