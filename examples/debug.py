import argparse

import torch

from accelerate.utils import convert_model, set_seed
from modeling_bert_te import BertForSequenceClassification as TEBertForSequenceClassification
from modeling_bert_te_lin import BertForSequenceClassification as TEBertForSequenceClassificationNoLN
from modeling_bert_te_ln import BertForSequenceClassification as TEBertForSequenceClassificationNoLinear
from transformer_engine.common.recipe import DelayedScaling
from transformer_engine.pytorch import fp8_autocast
from transformers import AutoTokenizer, BertForSequenceClassification


parser = argparse.ArgumentParser(description="Debugging conversion nn to te.")
parser.add_argument("--convert", action="store_true", help="Whether to convert or use the adapted models.")
parser.add_argument("--no_linear", action="store_true", help="Don't use te linear layers.")
parser.add_argument("--no_ln", action="store_true", help="Don't use te layernorm layers.")
args = parser.parse_args()

set_seed(42)
model = BertForSequenceClassification.from_pretrained("bert-base-cased").to(0).train()
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


inputs = tokenizer("Hello, my name is.", return_tensors="pt").to(0)
outputs = model(**inputs, labels=torch.tensor([0]).to(0))

state_dict = model.state_dict()

if args.convert:
    new_model = BertForSequenceClassification.from_pretrained("bert-base-cased").to(0).train()
    with torch.no_grad():
        convert_model(new_model)
else:
    if args.no_linear and args.no_ln:
        model_cls = BertForSequenceClassification
    elif args.no_linear:
        model_cls = TEBertForSequenceClassificationNoLinear
    elif args.no_ln:
        model_cls = TEBertForSequenceClassificationNoLN
    else:
        model_cls = TEBertForSequenceClassification
    new_model = model_cls.from_pretrained("bert-base-cased").to(0).train()

if not args.no_ln:
    state_dict = {k.replace("LayerNorm.", "LayerNorm.layer_norm_"): v for k, v in state_dict.items()}

new_model.load_state_dict(state_dict, strict=False)

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
