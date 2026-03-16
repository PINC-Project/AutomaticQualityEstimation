from bleurt import score

ref = "Czy może mi Pani powiedzieć jak dojechać do hotelu?"
tgt_1 = "Czy mogła by mi Pani wskazać drogę do hotelu?"
tgt_2 = "Czy może mi Pani powiedzieć jak dojechać do dworca?"
tgt_3 = "Modrzew to jedyne drzewo iglaste, które gubi igły w zimie."


candidates = [tgt_1, tgt_2, tgt_3]
references = [ref] * len(candidates)

bleurt = score.BleurtScorer()
scores = bleurt.score(references=references, candidates=candidates)

print(scores)
# outputs: [-0.056943848729133606, 0.2524346709251404, -0.9928008317947388]
# which is meh...
