from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")

inputs = tokenizer.encode(
    "Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy",
    return_tensors="pt",
)
outputs = model.generate(inputs)
# print(outputs)
print(tokenizer.decode(outputs[0]))
