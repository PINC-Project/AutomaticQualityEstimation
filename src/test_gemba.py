from gemba.utils import get_gemba_scores

src = "Can you please show me how to get to the hotel?"
tgt_1 = "Czy może mi Pani powiedzieć jak dojechać do hotelu?"
tgt_2 = "Czy może mi Pani powiedzieć jak dojechać do dworca?"
tgt_3 = "Modrzew to jedyne drzewo iglaste, które gubi igły w zimie."


hypothesis = [tgt_1, tgt_2, tgt_3]
source = [src] * len(hypothesis)


scores = get_gemba_scores(
    source=source,
    hypothesis=hypothesis,
    source_lang="English",
    target_lang="Polish",
    method="GEMBA-DA",
    model="gpt-4",
)

print(scores)
