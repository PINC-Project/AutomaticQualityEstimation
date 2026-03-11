import torch
from transformers import pipeline

source = [
    "I urge the House to give this the high importance that it deserves and to speak out for those in Uganda who currently are not being heard.",
    "Uganda is a deeply Christian country where traditional values hold sway.",
]

# model_id="Helsinki-NLP/opus-mt-pl-en"
# model_id = "pumad/pumadic-en-pl"
model_id = "sdadas/mt5-base-translator-en-pl"

pipeline = pipeline(
    "translation",
    model=model_id,
    dtype=torch.float16,
    device=0,
)

translation = pipeline(source)

print(translation)
