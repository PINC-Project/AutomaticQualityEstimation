import torch
from transformers import pipeline
import pandas as pd


def remove_disfluency(t: str) -> str:
    return " ".join(filter(lambda w: "~" not in w and "+" not in w, t.split()))


# model_plen = "Helsinki-NLP/opus-mt-pl-en"
# model_enpl = "pumad/pumadic-en-pl"

model_plen = "sdadas/mt5-base-translator-pl-en"
model_enpl = "sdadas/mt5-base-translator-en-pl"


df = pd.read_parquet("corpus.parquet")

sources_enpl = []
sources_enpl_idx = []
sources_plen = []
sources_plen_idx = []

for idx, row in df.iterrows():
    if row.UID[:2] == "EN":
        sources_enpl.append(remove_disfluency(row.source))
        sources_enpl_idx.append(idx)
    else:
        sources_plen.append(remove_disfluency(row.source))
        sources_plen_idx.append(idx)


pipe = pipeline(
    "translation",
    model=model_plen,
    dtype=torch.float16,
    device=0,
)
translation_plen = pipe(sources_plen, batch_size=8, max_length=512)
translation_plen = [x["translation_text"] for x in translation_plen]
df_plen = pd.DataFrame(
    index=sources_plen_idx, data=translation_plen, columns=["translation"]
)


pipe = pipeline(
    "translation",
    model=model_enpl,
    dtype=torch.float16,
    device=0,
)
translation_enpl = pipe(sources_enpl, batch_size=8, max_length=512)
translation_enpl = [x["translation_text"] for x in translation_enpl]

df_enpl = pd.DataFrame(
    index=sources_enpl_idx, data=translation_enpl, columns=["translation"]
)

df.update(df_plen)
df.update(df_enpl)

df.to_parquet("corpus_nmt_sdadas.parquet")
