import torch
from transformers import pipeline
import pandas as pd


def remove_disfluency(t: str) -> str:
    return " ".join(filter(lambda w: "~" not in w and "+" not in w, t.split()))


model_id = "facebook/nllb-200-3.3B"
src_lang = "pol_Latn"
tgt_lang = "eng_Latn"
# src_lang = "eng_Latn"
# tgt_lang = "pol_Latn"

# df = pd.read_parquet("corpus.parquet")
df = pd.read_parquet("corpus_nmt_nllb.parquet")

sources = []
sources_idx = []

for idx, row in df.iterrows():
    if row.UID[:2] != "EN":
        sources.append(remove_disfluency(row.source))
        sources_idx.append(idx)

translator = pipeline(
    "translation", model="facebook/nllb-200-3.3B", src_lang=src_lang, tgt_lang=tgt_lang
)
translation = translator(sources, batch_size=4, max_length=512)
translation = [x["translation_text"] for x in translation]
df_trans = pd.DataFrame(index=sources_idx, data=translation, columns=["translation"])
df.update(df_trans)

df.to_parquet("corpus_nmt_nllb2.parquet")
