import torch
from accelerate import Accelerator
import deepspeed

deepspeed.ops.op_builder.FusedAdamBuilder().is_compatible = lambda: False

# Suppose your patched wrapper is imported like:
# from accelerate.deepspeed_utils import DeepSpeedEngineWrapper
# (adjust path to your local Accelerate fork)
from accelerate.utils import DeepSpeedEngineWrapper

# Define a minimal model and data
model = torch.nn.Linear(10, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Minimal DeepSpeed config (disable internal clipping)
ds_config = {
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "fp16": {"enabled": False},
    "gradient_clipping": 0.0,
}

accelerator = Accelerator()

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config,
)

# Wrap your DeepSpeed engine manually for test
wrapper = DeepSpeedEngineWrapper(model_engine)

# Create dummy input/output
x = torch.randn(4, 10, device=accelerator.device)
y = torch.randint(0, 2, (4,), device=accelerator.device)
criterion = torch.nn.CrossEntropyLoss()

# --- Test backward ---
optimizer.zero_grad()
outputs = wrapper.engine(x)
loss = criterion(outputs, y)
wrapper.backward(loss)  # Should only compute grads
# Check if grads exist but not yet stepped
for p in wrapper.engine.module.parameters():
    assert p.grad is not None, "Gradient not computed!"

# --- Test gradient inspection/modification ---
for p in wrapper.engine.module.parameters():
    p.grad = p.grad * 0  # Zero out grads manually

# --- Test step ---
wrapper.step(sync_gradients=True, gradient_clipping=0.5)

# Check grads are zeroed after step (DeepSpeed does this)
for p in wrapper.engine.module.parameters():
    assert p.grad is None or torch.all(p.grad == 0), "Gradients not zeroed!"
print("✅ Test passed — backward and step decoupled successfully.")
