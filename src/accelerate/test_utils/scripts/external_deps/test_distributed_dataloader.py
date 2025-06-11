import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator


def main():
    accelerator = Accelerator()
    B, S, D = 2, 3, 4
    rank_data = torch.ones((B, S, D), device="cuda") * (accelerator.process_index + 1)
    all_rank_data = [torch.empty_like(rank_data) for _ in range(accelerator.num_processes)]
    torch.distributed.all_gather(all_rank_data, rank_data)

    dataloader = DataLoader(all_rank_data, batch_size=B, shuffle=False)
    dataloader = accelerator.prepare(dataloader)
    for batch in dataloader:
        all_rank_batch = [torch.empty_like(batch) for _ in range(accelerator.num_processes)]
        torch.distributed.all_gather(all_rank_batch, batch)

        if accelerator.is_main_process:
            for rank_idx in range(accelerator.num_processes):
                torch.testing.assert_close(
                    all_rank_batch[0],
                    all_rank_batch[rank_idx],
                    msg=f"Rank {rank_idx} batch {all_rank_batch[rank_idx]} differs from rank 0 batch {all_rank_batch[0]}",
                )

    accelerator.end_training()


if __name__ == "__main__":
    main()
