import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from rms.dataset import RM_Collator
from tqdm import tqdm
from torch.utils.data import Dataset


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn.modules.loss,
            gpu_id: int,
            save_every: int,
            save_dir: str
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.save_dir = save_dir

    def _run_batch(self, datas):
        self.optimizer.zero_grad()
        output = self.model(datas['input_ids'].to(self.gpu_id), datas['attention_mask'].to(self.gpu_id))
        loss = self.criterion(output, datas['labels'].to(self.gpu_id))
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data)))
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for datas in tqdm(self.train_data):
            self._run_batch(datas)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = os.path.join(self.save_dir, f"epoch{epoch}.pt")
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def prepare_dataloader(dataset: Dataset, batch_size: int, tokenizer, max_length: int):
    collator = RM_Collator(tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def do_train(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, dataset: Dataset, model, save_dir: str, lr: float, criterion: torch.nn.modules.loss, tokenizer, max_length: int):
    ddp_setup(rank, world_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_data = prepare_dataloader(dataset, batch_size, tokenizer, max_length)
    trainer = Trainer(model, train_data, optimizer, criterion, rank, save_every, save_dir)
    trainer.train(total_epochs)
    destroy_process_group()
