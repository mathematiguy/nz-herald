from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("models/nzherald-gpt2", local_files_only=True)

sentence = "Dragonfly Data Science"
input_ids = tokenizer.encode(sentence, return_tensors='pt')

output = model.generate(
    input_ids,
    do_sample=True,
    max_length=500,
    top_k=0,
    top_p=0.92,
    temperature=0.9)

print(tokenizer.decode(output[0], skip_special_tokens=True))
