from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model

src = "Can you please show me how to get to the hotel?"
tgt_1 = "Czy może mi Pani powiedzieć jak dojechać do hotelu?"
tgt_2 = "Czy może mi Pani powiedzieć jak dojechać do dworca?"
tgt_3 = "Modrzew to jedyne drzewo iglaste, które gubi igły w zimie."

blaser = load_blaser_model("blaser_2_0_qe").eval()
text_embedder = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder"
)

src_embs = text_embedder.predict([src, src, src], source_lang="eng_Latn")
mt_embs = text_embedder.predict([tgt_1, tgt_2, tgt_3], source_lang="pol_Latn")

print(blaser(src=src_embs, mt=mt_embs))
