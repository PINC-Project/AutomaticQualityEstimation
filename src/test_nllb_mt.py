from transformers import pipeline

source = [
    "I urge the House to give this the high importance that it deserves and to speak out for those in Uganda who currently are not being heard.",
    "Uganda is a deeply Christian country where traditional values hold sway.",
]

model_id = "facebook/nllb-200-3.3B"

translator = pipeline(
    "translation",
    model="facebook/nllb-200-3.3B",
    src_lang="eng_Latn",
    tgt_lang="pol_Latn",
)

translation = translator(source, max_length=512)
print(translation)
