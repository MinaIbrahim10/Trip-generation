import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, T5Tokenizer
model = TFAutoModelForSeq2SeqLM.from_pretrained("T5-Generation-Description-Model-Two")
tokenizer = T5Tokenizer.from_pretrained("T5-Generation-Description-Tokenazier")
new_keywords = input('enter trip location and place')
input_ids = tokenizer([new_keywords], return_tensors="tf").input_ids
generated_ids = model.generate(input_ids, max_length=163, num_beams=5, no_repeat_ngram_size=2)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)
