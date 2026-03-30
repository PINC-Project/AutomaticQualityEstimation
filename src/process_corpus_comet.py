import argparse
from pathlib import Path
import pandas as pd
from comet import download_model, load_from_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument("corpus", type=Path)
parser.add_argument("comet", type=Path)

args = parser.parse_args()

df = pd.read_parquet(str(args.corpus))


# model_path = download_model("Unbabel/wmt20-comet-qe-da")
model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
model = load_from_checkpoint(model_path)

data = [{"src": row.source, "mt": row.translation} for _, row in df.iterrows()]

model_output = model.predict(data, batch_size=4, gpus=1)

df["comet_score"] = model_output["scores"]

df.to_parquet(args.comet)
