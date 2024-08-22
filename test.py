import torch
from accelerate import Accelerator, DataLoaderConfiguration, skip_first_batches, load_checkpoint_in_model
from accelerate.utils import set_seed

from torchdata.stateful_dataloader import StatefulDataLoader
import tempfile

from torch.utils.data import DataLoader, TensorDataset

def create_dataloaders_for_test(batch_size=3, n_train_batches: int = 12, n_valid_batches: int = 2, num_workers=0):
    "Generates a tuple of dummy DataLoaders to test with"

    def get_dataset(n_batches):
        x = torch.randn(batch_size * n_batches, 3)
        y = torch.randn(batch_size * n_batches, 5)
        return TensorDataset(x, y)

    train_dataset = get_dataset(n_train_batches)
    valid_dataset = get_dataset(n_valid_batches)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
    return (train_dataloader, valid_dataloader)

class ModelForTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 4)
        self.batchnorm = torch.nn.BatchNorm1d(4)
        self.linear2 = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))

class ModelWithTiedWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.linear2 = torch.nn.Linear(4, 2)
        self.linear2.weight = self.linear1.weight
        self.linear2.bias = self.linear1.bias

    def forward(self, x):
        return self.linear2(self.linear1(x))


def create_components(tied_weights=False):
    model = ModelWithTiedWeights() if tied_weights else torch.nn.Linear(2, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=2, epochs=1)
    train_dl = DataLoader(TensorDataset(torch.tensor([1, 2, 3])))
    valid_dl = DataLoader(TensorDataset(torch.tensor([4, 5, 6])))
    return model, optimizer, scheduler, train_dl, valid_dl


set_seed(42)
dataloader_config = DataLoaderConfiguration(dispatch_batches=False, use_stateful_dataloader=True)
accelerator = Accelerator(dataloader_config=dataloader_config)

model, optimizer, scheduler, train_dl, valid_dl = create_components(False)
train_dl, valid_dl = create_dataloaders_for_test(n_train_batches=32)
model = ModelForTest()

(
    prepared_model,
    prepared_optimizer,
    prepared_scheduler,
    prepared_train_dl,
    prepared_valid_dl,
) = accelerator.prepare(model, optimizer, scheduler, train_dl, valid_dl)

assert isinstance(prepared_train_dl, StatefulDataLoader)
assert isinstance(prepared_valid_dl, StatefulDataLoader)

# Perform 4 training iterations to ensure the dataloader's iterator is advanced
# ~1/4 of the way there during DDP 2GPUs
# Should be 12 batches at the end
num_batches_to_skip = 4
model.train()
# path = "saved_model_for_test"
path = "saved_state_for_test"
not_skipped_batches = []
for step, batch in enumerate(prepared_train_dl):
    x, y = batch
    outputs = prepared_model(x)
    loss = torch.nn.functional.mse_loss(outputs, y)
    accelerator.backward(loss)
    prepared_optimizer.step()
    prepared_scheduler.step()
    prepared_optimizer.zero_grad()
    if step == num_batches_to_skip - 1:
        # state_dict = prepared_train_dl.state_dict()
        # # Save model for later use
        unwrapped_model = accelerator.unwrap_model(prepared_model)
        # accelerator.save_model(unwrapped_model, path)
        accelerator.save_state(path)
    if step >= num_batches_to_skip:
        not_skipped_batches.append(batch)

not_skipped_batches = accelerator.gather(not_skipped_batches)
# Should have 12 batches to go through still
assert len(not_skipped_batches) == 12, f"Skipped batches: {len(not_skipped_batches)}"

accelerator.wait_for_everyone()

original_linear1 = unwrapped_model.linear1.weight.clone()
original_batchnorm = unwrapped_model.batchnorm.weight.clone()
original_linear2 = unwrapped_model.linear2.weight.clone()

# Load the model and state dict
accelerator.load_state(path)
# load_checkpoint_in_model(prepared_model, path)
# set_seed(42)
# stateful_train_dl, _ = create_dataloaders_for_test(n_train_batches=32)
# prepared_stateful_train_dl = accelerator.prepare_data_loader(stateful_train_dl)
# prepared_stateful_train_dl.load_state_dict(state_dict)

# Train this to the end of the DataLoader
batches_seen_with_loaded_dl = 0
for batch in prepared_train_dl:
    x, y = batch
    outputs = prepared_model(x)
    loss = torch.nn.functional.mse_loss(outputs, y)
    accelerator.backward(loss)
    prepared_optimizer.step()
    prepared_scheduler.step()
    prepared_optimizer.zero_grad()
    batches_seen_with_loaded_dl += 1

unwrapped_model_2 = accelerator.unwrap_model(prepared_model)

new_linear1 = unwrapped_model_2.linear1.weight
new_batchnorm = unwrapped_model_2.batchnorm.weight
new_linear2 = unwrapped_model_2.linear2.weight

# Assert equalities
assert batches_seen_with_loaded_dl == 12, f"Batches seen: {batches_seen_with_loaded_dl}"
assert torch.allclose(original_linear1, new_linear1)
assert torch.allclose(original_batchnorm, new_batchnorm)
assert torch.allclose(original_linear2, new_linear2)