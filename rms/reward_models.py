from abc import ABC
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from rms.ddp_utils import *
import torch.multiprocessing as mp
from rms.transformers_utils import *
from typing import Union
import deepspeed


class BaseRM(ABC):
    def __init__(self, backend: str = None, rm_type: str = 'prm'):
        self.backend = backend
        self.rm = None
        assert rm_type in ['prm', 'orm'], "rm_type must be either 'prm' or 'orm'\n"
        self.rm_type = rm_type

    def install(self):
        raise NotImplementedError("The method 'install' must be implemented for a rm\n")

    def eval(self, q: str, state: str) -> float:
        raise NotImplementedError("The method 'eval' must be implemented for a rm\n")

    def eval_batch(self, q: str, states: list[str]) -> list[float]:
        raise NotImplementedError("The method 'eval_batch' must be implemented for a rm\n")


class CoreScalarRM(nn.Module):
    def __init__(self, backend: str):
        super(CoreScalarRM, self).__init__()
        self.backend = backend
        self.config = AutoConfig.from_pretrained(backend, trust_remote_code=True)
        self.base_model = AutoModelForCausalLM.from_pretrained(backend, trust_remote_code=True,
                                                               torch_dtype=self.config.torch_dtype)
        self.vocab_size = self.base_model.config.vocab_size
        self.dtype = self.config.torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(backend, trust_remote_code=True)
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
            print("Warning: setting tokenizer padding_side to 'left'\n")
        if self.tokenizer.truncation_side != "left":
            self.tokenizer.truncation_side = "left"
            print("Warning: setting tokenizer truncation_side to 'left'\n")
        if self.tokenizer.pad_token is None:
            print("No pad token found, setting pad token to eos token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.base_model.config.pad_token_id = self.base_model.config.eos_token_id
        self.ln = nn.Linear(self.vocab_size, 1, dtype=self.dtype)

    def forward(self, input_ids, attention_mask):
        logits = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]
        logits = logits.to(dtype=self.dtype)
        rws = self.ln(logits)
        return rws.squeeze(dim=1).to(dtype=torch.float32)


class StandardScalarRM(BaseRM):
    """
    Standard reward model class
    Input: str
    Output: scalar
    """

    def __init__(self, backend: str = None, rm_type: str = 'prm', device: str = 'cuda', state_pth: str = None,
                 max_length: int = 1024, min_rw: float = 0.0, max_rw: float = 1.0):
        super().__init__(backend, rm_type)
        self.rm_class = "standard_scalar"
        self.device = device
        self.state_pth = state_pth
        self.install()
        assert self.rm is not None, "The rm has not been correctly installed\n"
        self.tokenizer = self.rm.tokenizer
        self.max_length = max_length
        self.min_rw = min_rw
        self.max_rw = max_rw

    def install(self):
        self.rm = CoreScalarRM(self.backend)
        if self.state_pth is not None:
            self.rm.load_state_dict(torch.load(self.state_pth))
        self.rm.to(self.device)

    def encode(self, q: str, state: str) -> dict:
        input_str = input_format.format(p=q, a=state)
        encoded_pair = self.tokenizer(
            input_str,
            padding=True,
            max_length=self.max_length,  # Set the max length
            truncation=True,
            return_tensors='pt',  # Return PyTorch Tensor format
        )

        return {
            'input_ids': encoded_pair['input_ids'].to(self.device),
            'attention_mask': encoded_pair['attention_mask'].to(self.device),
        }

    def encode_batch(self, q: str, states: list[str]) -> dict:
        input_strs = [input_format.format(p=q, a=state) for state in states]
        encoded_pairs = self.tokenizer(
            input_strs,
            padding=True,
            max_length=self.max_length,  # Set the max length
            truncation=True,
            return_tensors='pt',  # Return PyTorch Tensor format
        )

        return {
            'input_ids': encoded_pairs['input_ids'].to(self.device),
            'attention_mask': encoded_pairs['attention_mask'].to(self.device),
        }

    def eval(self, q: str, state: str) -> float:
        inputs = self.encode(q, state)
        with torch.no_grad():
            rw = self.rm(**inputs).item()
            rw = max(min(rw, self.max_rw), self.min_rw)
            return rw

    def eval_batch(self, q: str, states: list[str]) -> list[float]:
        all_rws = []
        batch_size = 5
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i + batch_size]
            inputs = self.encode_batch(q, batch_states)
            with torch.no_grad():
                rws = self.rm(**inputs)
                rws = rws.tolist()
                rws = [max(min(rw, self.max_rw), self.min_rw) for rw in rws]
                all_rws.extend(rws)
        return all_rws

    def train(self, train_pth: str, test_pth: str = None, batch_size: int = 16, lr: float = 1e-4,
              epochs: int = 1, save_dir: str = None, save_every: int = 1, test_tol: float = 0.05):

        # prepare data
        train_data = RM_Dataset(train_pth)
        do_test = True if test_pth is not None else False
        test_data = RM_Dataset(test_pth) if do_test else None

        # train
        criterion = nn.MSELoss(reduction='sum')
        optimizer = AdamW(self.rm.parameters(), lr=lr)
        train_losses = []
        collator = RM_Collator(self.tokenizer, max_length=self.max_length)
        train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collator, shuffle=True)
        print("Start training...\n")
        for epoch in range(epochs):
            print(f"Epoch <{epoch + 1}/{epochs}> training...\n")

            train_loss = 0.0
            self.rm.train()
            for batch in tqdm(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.rm(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_data)
            train_losses.append(avg_train_loss)

            print(f"Epoch <{epoch + 1}> ends...\n")
            print(f"Train Loss: {avg_train_loss}\n")
            if epoch % save_every == 0:
                save_pth = os.path.join(save_dir, f"epoch{epoch}.pt")
                torch.save(self.rm.state_dict(), save_pth)

        # test
        test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collator) if do_test else None
        if do_test:
            print("Start testing...\n")
            result_items = []
            for epoch in range(epochs):
                if epoch % save_every == 0:
                    model_pth = os.path.join(save_dir, f"epoch{epoch}.pt")
                    print(f"Testing model <{model_pth}>\n")
                    self.rm.load_state_dict(torch.load(model_pth))
                    test_preds = []
                    test_labels = []
                    with torch.no_grad():
                        self.rm.eval()
                        for batch in tqdm(test_loader):
                            input_ids = batch['input_ids'].to(self.device)
                            attention_mask = batch['attention_mask'].to(self.device)
                            labels = batch['labels'].to(self.device)
                            outputs = self.rm(input_ids=input_ids, attention_mask=attention_mask)
                            test_preds.extend(outputs.tolist())
                            test_labels.extend(labels.tolist())

                    print("Test ends, showing prediction results...\n")
                    passed = 0
                    total = 0
                    for pred, label in zip(test_preds, test_labels):
                        print(f"Pred: {pred}, Label: {label}")
                        total += 1
                        if abs(min(self.max_rw, max(self.min_rw, pred)) - label) < test_tol:
                            passed += 1
                    print(f"\n<Test result>\nPassed: {passed}, Total: {total}, Pass Rate: {passed / total}\n")
                    result_item = {'model': model_pth,
                                   'result': {'passed': passed, 'total': total, 'pass_rate': passed * 1.0 / total}}
                    result_items.append(result_item)

            return result_items

        else:
            return None

    def ddp_train(self, train_pth: str, test_pth: str = None, batch_size: int = 16, lr: float = 1e-4,
                  epochs: int = 1, save_dir: str = None, save_every: int = 1, test_tol: float = 0.05):

        # setup ddp
        world_size = torch.cuda.device_count()
        print(f"world size: {world_size}\n")

        # prepare data
        train_data = RM_Dataset(train_pth)
        do_test = True if test_pth is not None else False
        test_data = RM_Dataset(test_pth) if do_test else None

        # train
        print("Start training...\n")
        criterion = nn.MSELoss(reduction='sum')
        mp.spawn(do_train,
                 args=(world_size, save_every, epochs, batch_size, train_data, self.rm, save_dir, lr, criterion,
                       self.tokenizer, self.max_length),
                 nprocs=world_size)

        # test
        collator = RM_Collator(self.tokenizer, max_length=self.max_length)
        test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collator) if do_test else None
        if do_test:
            print("Start testing...\n")
            result_items = []
            for epoch in range(epochs):
                if epoch % save_every == 0:
                    model_pth = os.path.join(save_dir, f"epoch{epoch}.pt")
                    print(f"Testing model <{model_pth}>\n")
                    self.rm.load_state_dict(torch.load(model_pth))
                    self.rm.to(self.device)
                    test_preds = []
                    test_labels = []
                    with torch.no_grad():
                        self.rm.eval()
                        for batch in tqdm(test_loader):
                            input_ids = batch['input_ids'].to(self.device)
                            attention_mask = batch['attention_mask'].to(self.device)
                            labels = batch['labels'].to(self.device)
                            outputs = self.rm(input_ids=input_ids, attention_mask=attention_mask)
                            test_preds.extend(outputs.tolist())
                            test_labels.extend(labels.tolist())

                    print("Test ends, showing prediction results...\n")
                    passed = 0
                    total = 0
                    for pred, label in zip(test_preds, test_labels):
                        print(f"Pred: {pred}, Label: {label}")
                        total += 1
                        if abs(min(self.max_rw, max(self.min_rw, pred)) - label) < test_tol:
                            passed += 1
                    print(f"\n<Test result>\nPassed: {passed}, Total: {total}, Pass Rate: {passed / total}\n")
                    result_item = {'model': model_pth,
                                   'result': {'passed': passed, 'total': total, 'pass_rate': passed * 1.0 / total}}
                    result_items.append(result_item)

            return result_items

        else:
            return None


class TransformersScalarRM(BaseRM):
    """
    Transformers sequence classification model based reward model class
    Input: str
    Output: scalar
    """

    def __init__(self, backend: str = None, rm_type: str = 'prm', stage: str = 'test', max_length: int = 1024,
                 min_rw: float = 0.0, max_rw: float = 1.0, device: Union[str, torch.device] = 'cuda'):
        super().__init__(backend, rm_type)
        assert stage in ['test', 'train'], "The stage should be either 'test' or 'train'\n"
        self.device = device
        self.stage = stage
        self.rm_class = "transformers_scalar"
        self.max_length = max_length
        self.min_rw = min_rw
        self.max_rw = max_rw
        self.config = AutoConfig.from_pretrained(self.backend, trust_remote_code=True)
        self.tokenizer = None
        self.rm = None
        if self.stage == 'test':
            self.install()
            assert self.rm is not None, "The rm has not been correctly installed\n"
            assert self.tokenizer is not None, "The tokenizer has not been correctly installed\n"
            self.rm.eval()

    def install(self):
        model_pth = self.backend
        self.tokenizer = AutoTokenizer.from_pretrained(model_pth, trust_remote_code=True)
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
            print("Warning: setting tokenizer padding_side to 'left'\n")
        if self.tokenizer.truncation_side != "left":
            self.tokenizer.truncation_side = "left"
            print("Warning: setting tokenizer truncation_side to 'left'\n")
        if self.stage == 'test':
            self.rm = AutoModelForSequenceClassification.from_pretrained(model_pth,
                                                                         torch_dtype=self.config.torch_dtype,
                                                                         num_labels=1, trust_remote_code=True).to(self.device, dtype=self.config.torch_dtype)
        else:
            with deepspeed.zero.Init():
                self.rm = AutoModelForSequenceClassification.from_pretrained(model_pth,
                                                                             torch_dtype=self.config.torch_dtype,
                                                                             num_labels=1, trust_remote_code=True).to(self.device, dtype=self.config.torch_dtype)
        if self.tokenizer.pad_token is None:
            print("No pad token found, setting pad token to eos token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.rm.config.pad_token_id = self.rm.config.eos_token_id

    def encode(self, q: str, state: str) -> dict:
        input_str = input_format.format(p=q, a=state)
        encoded_pair = self.tokenizer(
            input_str,
            padding=True,
            max_length=self.max_length,  # Set the max length
            truncation=True,
            return_tensors='pt',  # Return PyTorch Tensor format
        )

        return {
            'input_ids': encoded_pair['input_ids'].to(device=self.device),
            'attention_mask': encoded_pair['attention_mask'].to(device=self.device),
        }

    def encode_batch(self, q: str, states: list[str]) -> dict:
        input_strs = [input_format.format(p=q, a=state) for state in states]
        encoded_pair = self.tokenizer(
            input_strs,
            padding=True,
            max_length=self.max_length,  # Set the max length
            truncation=True,
            return_tensors='pt',  # Return PyTorch Tensor format
        )

        return {
            'input_ids': encoded_pair['input_ids'].to(device=self.device),
            'attention_mask': encoded_pair['attention_mask'].to(device=self.device),
        }

    def eval(self, q: str, state: str) -> float:
        assert self.stage == 'test', "The model is not in test mode\n"
        inputs = self.encode(q, state)
        with torch.no_grad():
            outputs = self.rm(**inputs)
            rw = outputs.get('logits').squeeze().to(dtype=torch.float32).item()
            rw = max(min(rw, self.max_rw), self.min_rw)
            return rw

    def eval_batch(self, q: str, states: list[str]) -> list[float]:
        assert self.stage == 'test', "The model is not in test mode\n"
        all_rws = []
        batch_size = 5
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i + batch_size]
            inputs = self.encode_batch(q, batch_states)
            with torch.no_grad():
                outputs = self.rm(**inputs)
                rws = outputs.get('logits').squeeze(dim=1).to(dtype=torch.float32).tolist()
                rws = [max(min(rw, self.max_rw), self.min_rw) for rw in rws]
                all_rws.extend(rws)
        return all_rws

    def train(self, args):
        assert self.stage == 'train', "The model is not in train mode\n"
        set_test_tol(args.test_tol)
        train_args = set_training_args(args)
        self.install()
        assert self.rm is not None, "The rm has not been correctly installed\n"
        assert self.tokenizer is not None, "The tokenizer has not been correctly installed\n"
        train_set, test_set = load_datasets(args)
        collator = RM_Collator(self.tokenizer, max_length=self.max_length)
        trainer = RMTrainer(model=self.rm, args=train_args, data_collator=collator, train_dataset=train_set,
                            eval_dataset=test_set,
                            compute_metrics=compute_metrics_scalar, rm_class=self.rm_class)

        # train and evaluate
        trainer.train()
        test_metrics = trainer.evaluate()

        # save model and tokenizer
        save_best_dir = os.path.join(args.save_dir, 'best')
        print("Saving best model to {}\n".format(save_best_dir))
        trainer.save_model(save_best_dir)
        self.tokenizer.save_pretrained(save_best_dir)

        # output results
        result_item = {'model': save_best_dir, 'result': test_metrics}
        return result_item


class TransformersProbRM(BaseRM):
    """
    Transformers sequence classification model based reward model class, use probs as final output
    Input: str
    Output: prob, ranged between 0 and 1
    """

    def __init__(self, backend: str = None, rm_type: str = 'prm', stage: str = 'test', max_length: int = 1024,
                 device: Union[str, torch.device] = 'cuda'):
        super().__init__(backend, rm_type)
        assert stage in ['test', 'train'], "The stage should be either 'test' or 'train'\n"
        self.device = device
        self.stage = stage
        self.rm_class = "transformers_prob"
        self.max_length = max_length
        self.config = AutoConfig.from_pretrained(self.backend, trust_remote_code=True)
        self.tokenizer = None
        self.rm = None
        if self.stage == 'test':
            self.install()
            assert self.rm is not None, "The rm has not been correctly installed\n"
            assert self.tokenizer is not None, "The tokenizer has not been correctly installed\n"
            self.rm.eval()

    def install(self):
        model_pth = self.backend
        self.tokenizer = AutoTokenizer.from_pretrained(model_pth, trust_remote_code=True)
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
            print("Warning: setting tokenizer padding_side to 'left'\n")
        if self.tokenizer.truncation_side != "left":
            self.tokenizer.truncation_side = "left"
            print("Warning: setting tokenizer truncation_side to 'left'\n")
        if self.stage == 'test':
            self.rm = AutoModelForSequenceClassification.from_pretrained(model_pth,
                                                                         torch_dtype=self.config.torch_dtype,
                                                                         num_labels=2, trust_remote_code=True).to(self.device, dtype=self.config.torch_dtype)
        else:
            with deepspeed.zero.Init():
                self.rm = AutoModelForSequenceClassification.from_pretrained(model_pth,
                                                                             torch_dtype=self.config.torch_dtype,
                                                                             num_labels=2, trust_remote_code=True).to(self.device, dtype=self.config.torch_dtype)
        if self.tokenizer.pad_token is None:
            print("No pad token found, setting pad token to eos token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.rm.config.pad_token_id = self.rm.config.eos_token_id

    def encode(self, q: str, state: str) -> dict:
        input_str = input_format.format(p=q, a=state)
        encoded_pair = self.tokenizer(
            input_str,
            padding=True,
            max_length=self.max_length,  # Set the max length
            truncation=True,
            return_tensors='pt',  # Return PyTorch Tensor format
        )

        return {
            'input_ids': encoded_pair['input_ids'].to(device=self.device),
            'attention_mask': encoded_pair['attention_mask'].to(device=self.device),
        }

    def encode_batch(self, q: str, states: list[str]) -> dict:
        input_strs = [input_format.format(p=q, a=state) for state in states]
        encoded_pair = self.tokenizer(
            input_strs,
            padding=True,
            max_length=self.max_length,  # Set the max length
            truncation=True,
            return_tensors='pt',  # Return PyTorch Tensor format
        )

        return {
            'input_ids': encoded_pair['input_ids'].to(device=self.device),
            'attention_mask': encoded_pair['attention_mask'].to(device=self.device),
        }

    def eval(self, q: str, state: str) -> float:
        assert self.stage == 'test', "The model is not in test mode\n"
        inputs = self.encode(q, state)
        with torch.no_grad():
            outputs = self.rm(**inputs)
            logits = outputs.get('logits').to(dtype=torch.float32).squeeze()
            probs = F.softmax(logits)
            rw = probs[1].item()
            return rw

    def eval_batch(self, q: str, states: list[str]) -> list[float]:
        assert self.stage == 'test', "The model is not in test mode\n"
        all_rws = []
        batch_size = 5
        for i in range(0, len(states), batch_size):
            batch_states = states[i:i + batch_size]
            inputs = self.encode_batch(q, batch_states)
            with torch.no_grad():
                outputs = self.rm(**inputs)
                logits = outputs.get('logits').to(dtype=torch.float32)
                probs = F.softmax(logits, dim=1)
                rws = probs[:, 1].tolist()
                all_rws.extend(rws)
        return all_rws

    def train(self, args):
        assert self.stage == 'train', "The model is not in train mode\n"
        set_test_tol(args.test_tol)
        set_prob_loss_weight(args.loss_weight_factor)
        train_args = set_training_args(args)
        self.install()
        assert self.rm is not None, "The rm has not been correctly installed\n"
        assert self.tokenizer is not None, "The tokenizer has not been correctly installed\n"
        train_set, test_set = load_datasets(args)
        collator = RM_Collator(self.tokenizer, max_length=self.max_length)
        trainer = RMTrainer(model=self.rm, args=train_args, data_collator=collator, train_dataset=train_set,
                            eval_dataset=test_set,
                            compute_metrics=compute_metrics_prob, rm_class=self.rm_class)

        # train and evaluate
        trainer.train()
        test_metrics = trainer.evaluate()

        # save model and tokenizer
        save_best_dir = os.path.join(args.save_dir, 'best')
        print("Saving best model to {}\n".format(save_best_dir))
        trainer.save_model(save_best_dir)
        self.tokenizer.save_pretrained(save_best_dir)

        # output results
        result_item = {'model': save_best_dir, 'result': test_metrics}
        return result_item
