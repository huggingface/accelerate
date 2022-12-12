from accelerate.utils import convert_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased").to(0)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


inputs = tokenizer("Hello, my name is.", return_tensors="pt").to(0)
outputs = model(**inputs)

convert_model(model)
new_outputs = model(**inputs)

print(outputs, new_outputs)
print(torch.allclose(outputs.logits, new_outputs.logits))