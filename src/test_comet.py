from comet import download_model, load_from_checkpoint

src = "Can you please show me how to get to the hotel?"
tgt_1 = "Czy może mi Pani powiedzieć jak dojechać do hotelu?"
tgt_2 = "Czy może mi Pani powiedzieć jak dojechać do dworca?"
tgt_3 = "Modrzew to jedyne drzewo iglaste, które gubi igły w zimie."

# model_path = download_model("Unbabel/wmt20-comet-qe-da")
model_path = download_model("Unbabel/wmt23-cometkiwi-da-xl")
model = load_from_checkpoint(model_path)

data = [{"src": src, "mt": tgt} for tgt in (tgt_1, tgt_2, tgt_3)]

model_output = model.predict(data, batch_size=1, gpus=1)

print(model_output)
