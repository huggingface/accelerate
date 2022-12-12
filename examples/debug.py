from accelerate.utils import convert_model, set_seed
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformer_engine.pytorch import fp8_autocast
from transformer_engine.common.recipe import DelayedScaling

set_seed(42)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased").to(0).train()
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


inputs = tokenizer("Hello, my name is.", return_tensors="pt").to(0)
outputs = model(**inputs, labels=torch.tensor([0]).to(0))

set_seed(42)
new_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased").to(0).train()
with torch.no_grad():
    convert_model(new_model)
# new_model.forward = fp8_autocast(enabled=False, fp8_recipe=DelayedScaling())(new_model.forward)
new_outputs = new_model(**inputs, labels=torch.tensor([0]).to(0))

print("Outputs comparison at 1e-6/1e-5/1e-4")
print(torch.allclose(outputs.logits, new_outputs.logits, atol=1e-6))
print(torch.allclose(outputs.logits, new_outputs.logits, atol=1e-5))
print(torch.allclose(outputs.logits, new_outputs.logits, atol=1e-4))

outputs.loss.backward()
new_outputs.loss.backward()

grad1 = model.bert.embeddings.word_embeddings.weight.grad
grad2 = model.bert.embeddings.word_embeddings.weight.grad
print("Embeddings gradients at 1e-6/1e-5/1e-4")
print(torch.allclose(grad1, grad2, atol=1e-6))
print(torch.allclose(grad1, grad2, atol=1e-5))
print(torch.allclose(grad1, grad2, atol=1e-4))

grad1 = getattr(model.bert.encoder.layer, "0").attention.self.query.weight.grad
grad2 = getattr(new_model.bert.encoder.layer, "0").attention.self.query.weight.grad
print("Linear gradients at 1e-6/1e-5/1e-4")
print(torch.allclose(grad1, grad2, atol=1e-6))
print(torch.allclose(grad1, grad2, atol=1e-5))
print(torch.allclose(grad1, grad2, atol=1e-4))

grad1 = getattr(model.bert.encoder.layer, "0").attention.output.LayerNorm.weight.grad
grad2 = getattr(new_model.bert.encoder.layer, "0").attention.output.LayerNorm.layer_norm_weight.grad
print("Layer norm gradients at 1e-6/1e-5/1e-4")
print(torch.allclose(grad1, grad2, atol=1e-6))
print(torch.allclose(grad1, grad2, atol=1e-5))
print(torch.allclose(grad1, grad2, atol=1e-4))

