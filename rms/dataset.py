import torch
from torch.utils.data import Dataset
from utils.json_operator import *
from rms.input import *


class RM_Dataset(Dataset):
    """
    Dataset class for reward model training
    Loads data from jsonl files
    Keys: prompt, answer, reward
    """

    def __init__(self, data_file: str):
        self.file_path = data_file
        self.datas = read_json(self.file_path)
        assert len(self.datas) > 0, f"No data found in the dataset file: {self.file_path}\n"
        assert self.key_check(), "Error: keys not found in data!\n"

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        prompt = data['prompt']
        answer = data['answer']
        reward = data['reward']
        input_str = input_format.format(p=prompt, a=answer)

        return {
            'input_str': input_str,
            'label': reward
        }

    def key_check(self):
        ok = True
        keys = ['prompt', 'answer', 'reward']
        for key in keys:
            if key not in self.datas[0]:
                print(f'Error: key "{key}" not found in data!\n')
                ok = False
                break
        return ok


class RM_Collator:
    """
    DataCollator class for reward model training
    Notice: tokenizer padding_side and truncation_side should be 'left'
    """

    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Ensure tokenizer settings
        assert self.tokenizer.padding_side == 'left', "Tokenizer padding_side must be 'left'.\n"
        assert self.tokenizer.truncation_side == 'left', "Tokenizer truncation_side must be 'left'.\n"

    def __call__(self, batch):
        input_strs = [item['input_str'] for item in batch]
        labels = [item['label'] for item in batch]

        encoded_batch = self.tokenizer(
            input_strs,
            padding=True,
            max_length=self.max_length,  # Set the max length
            truncation=True,
            return_tensors='pt',  # Return PyTorch Tensor format
        )

        return {
            'input_ids': encoded_batch['input_ids'],
            'attention_mask': encoded_batch['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.float32)
        }
