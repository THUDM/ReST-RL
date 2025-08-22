import torch.nn as nn
import datasets
import torch.nn.functional as F
from torch.utils.data import random_split, IterableDataset
import numpy as np
from rms.dataset import *
from utils.json_operator import *
from transformers import Trainer, TrainingArguments, DataCollator, PreTrainedModel, TrainerCallback, EvalPrediction
from typing import Union, Tuple, Optional, List, Callable, Dict

TEST_TOL = 0.05
PROB_LOSS_WEIGHT = 0.5


def set_test_tol(tol: float):
    global TEST_TOL
    TEST_TOL = tol


def set_prob_loss_weight(weight: float):
    global PROB_LOSS_WEIGHT
    if weight < 0.0 or weight > 1.0:
        print('Error: loss weight factor should be in range [0, 1]. Default weight factor will be used\n')
    else:
        PROB_LOSS_WEIGHT = weight
        print(f"Warning: loss weight factor is set to {PROB_LOSS_WEIGHT}\n")


def load_deepspeed_config(args):
    config_dict = json.load(open(args.deepspeed_config))
    if 'optimizer' in config_dict:
        if config_dict['optimizer']['params']['lr'] != 'auto':
            config_dict['optimizer']['params']['lr'] = args.lr
        if config_dict['optimizer']['params']['weight_decay'] != 'auto':
            config_dict['optimizer']['params']['weight_decay'] = args.weight_decay
    if config_dict['gradient_accumulation_steps'] != 'auto':
        config_dict['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if config_dict['train_micro_batch_size_per_gpu'] != 'auto':
        config_dict['train_micro_batch_size_per_gpu'] = args.batch_size_per_device
    return config_dict


def split_dataset(dataset: RM_Dataset):
    train_ratio = 0.95
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    if test_size < 100:
        print('Test size is too small, set to 100 by default\n')
        test_size = 100
        train_size = len(dataset) - test_size
        assert train_size > test_size, "Dataset is too small, please enlarge the dataset to ensure valid training results\n"
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


def load_datasets(args):
    train_data_pth = args.train_pth
    test_data_pth = args.test_pth

    if test_data_pth is not None:
        train_set = RM_Dataset(train_data_pth)
        test_set = RM_Dataset(test_data_pth)
    else:
        dataset = RM_Dataset(train_data_pth)
        train_set, test_set = split_dataset(dataset)
    return train_set, test_set


def set_training_args(args):
    deepspeed_config = load_deepspeed_config(args)
    output_dir = args.save_dir

    training_arg = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.batch_size_per_device,
        per_device_eval_batch_size=args.batch_size_per_device,
        num_train_epochs=args.epochs,
        save_strategy='steps',
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_only_model=args.save_only_model,
        eval_strategy='steps',
        eval_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        ddp_backend='nccl',
        deepspeed=deepspeed_config,
        remove_unused_columns=False,
        report_to=args.report_to,
    )
    return training_arg


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def compute_metrics_prob(pred):
    labels = pred.label_ids
    preds_ = pred.predictions
    preds = softmax(preds_)[:, 1]
    acc = (np.abs(labels - preds) < TEST_TOL).mean()
    return {"accuracy": acc}


def compute_metrics_scalar(pred):
    labels = pred.label_ids
    preds_ = pred.predictions
    preds = np.clip(preds_[:, 0], 0, 1)
    acc = (np.abs(labels - preds) < TEST_TOL).mean()
    return {"accuracy": acc}


def compute_loss_prob(model, inputs, return_outputs=False):
    labels_ = inputs.get('labels').to(model.device)
    label_cls1 = 1.0 - labels_
    labels = torch.stack([label_cls1, labels_], dim=1)
    input_ids = inputs.get('input_ids').to(model.device)
    attention_mask = inputs.get('attention_mask').to(model.device)
    outputs = model(input_ids, attention_mask)
    logits = outputs.get('logits').to(dtype=torch.float32)
    loss = F.cross_entropy(logits, labels, weight=torch.tensor([1.0 - PROB_LOSS_WEIGHT, PROB_LOSS_WEIGHT], device=model.device))
    return (loss, outputs) if return_outputs else loss


def compute_loss_scalar(model, inputs, return_outputs=False):
    labels = inputs.get('labels').to(model.device)
    input_ids = inputs.get('input_ids').to(model.device)
    attention_mask = inputs.get('attention_mask').to(model.device)
    outputs = model(input_ids, attention_mask)
    logits = outputs.get('logits').squeeze(-1).to(dtype=torch.float32)
    loss = F.mse_loss(logits, labels)
    return (loss, outputs) if return_outputs else loss


def show_preds(preds, labels):
    if type(preds) is not list:
        preds_ = [preds]
    else:
        preds_ = preds
    if type(labels) is not list:
        labels_ = [labels]
    else:
        labels_ = labels

    for i in range(len(preds_)):
        print(f"Sample {i + 1}: Prediction Score: {preds_[i]:.4f}, Actual Score: {labels_[i]:.4f}")


class RMTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], "datasets.Dataset"]] = None,
            model_init: Optional[Callable[[], PreTrainedModel]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
            rm_class: str = None
    ):
        super().__init__(model=model, args=args, data_collator=data_collator, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, model_init=model_init,
                         compute_metrics=compute_metrics, callbacks=callbacks, optimizers=optimizers,
                         preprocess_logits_for_metrics=preprocess_logits_for_metrics)
        assert rm_class in ['transformers_scalar', 'transformers_prob'], "Reward model class must be one of 'transformers_scalar', 'transformers_prob'\n"
        self.rm_class = rm_class

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.rm_class == 'transformers_scalar':
            return compute_loss_scalar(model, inputs, return_outputs)
        else:
            return compute_loss_prob(model, inputs, return_outputs)
