import unittest
import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.data_loader import DataLoaderShard

class SimpleDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
    def __getitem__(self, index):
        return torch.tensor([index])
    def __len__(self):
        self.size
        return self.size

class DataLoaderWithoutBatchSampler(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'batch_sampler'):
            delattr(self, 'batch_sampler')

class AccelerateIssue882Tests(unittest.TestCase):
    def test_prepare_dataloader_without_batch_sampler(self):
        accelerator = Accelerator()
        dataset = SimpleDataset()
        dataloader = DataLoaderWithoutBatchSampler(dataset, batch_size=32)
        
        # This should not raise AttributeError
        prepared_dl = accelerator.prepare(dataloader)
        
        self.assertIsInstance(prepared_dl, (DataLoader, DataLoaderShard))
        # Verify it can still be iterated
        batch = next(iter(prepared_dl))
        self.assertEqual(batch.shape[0], 32)

if __name__ == "__main__":
    unittest.main()
