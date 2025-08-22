import torch
from utils.json_operator import *
from utils.dataset_operator import *
from transformers import AutoTokenizer, AutoConfig

from trl import (
    DPOConfig,
    DPOTrainer,
)


def check_dpo_dataset_format(dataset: list[dict]):
    data = dataset[0]
    assert type(data) == dict, "The dataset should be a list of dict\n"
    assert 'prompt' in data and 'chosen' in data and 'rejected' in data, "The dataset should be in standard preference format\n"


def load_dpo_datasets(args):
    train_data_pth = args.train_pth
    test_data_pth = args.test_pth
    train_data_list = read_json(train_data_pth)
    check_dpo_dataset_format(train_data_list)
    train_set = Dataset.from_list(train_data_list)
    train_set = train_set.shuffle(seed=42)

    if test_data_pth is not None:
        test_data_list = read_json(test_data_pth)
        check_dpo_dataset_format(test_data_list)
        test_set = Dataset.from_list(test_data_list)

    else:
        test_set = None

    return train_set, test_set


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


def set_dpo_training_args(args, model_config):
    deepspeed_config = load_deepspeed_config(args)
    output_dir = args.save_dir
    do_eval = args.test_pth is not None

    training_arg = DPOConfig(
        output_dir=output_dir,
        do_train=True,
        do_eval=do_eval,
        per_device_train_batch_size=args.batch_size_per_device,
        per_device_eval_batch_size=args.batch_size_per_device,
        num_train_epochs=args.epochs,
        save_strategy='steps',
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_only_model=args.save_only_model,
        eval_strategy='steps' if do_eval else 'no',
        eval_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        ddp_backend='nccl',
        load_best_model_at_end=True if do_eval else False,
        deepspeed=deepspeed_config,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        report_to=args.report_to,
        bf16=True if model_config.torch_dtype == torch.bfloat16 else False,
        fp16=True if model_config.torch_dtype == torch.float16 else False,
        model_init_kwargs={"trust_remote_code": True, "torch_dtype": model_config.torch_dtype},
        ref_model_init_kwargs={"trust_remote_code": True, "torch_dtype": model_config.torch_dtype},
    )
    return training_arg, do_eval


def dpo_train(args):
    """
    Direct Preference Optimization with trl. Your dataset must use the standard preference format.
    :param args: training args
    :return: test results
    """

    ################
    # Model Config
    ################
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print("Using torch dtype: ", model_config.torch_dtype, '\n')

    ################
    # Train Args
    ################
    dpo_config, do_eval = set_dpo_training_args(args, model_config)

    ################
    # Init Tokenizer
    ################
    processing_class = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ################
    # Dataset
    ################
    train_set, test_set = load_dpo_datasets(args)

    ################
    # Training
    ################
    trainer = DPOTrainer(
        model=args.model,
        ref_model=args.model,
        args=dpo_config,
        train_dataset=train_set,
        eval_dataset=test_set,
        processing_class=processing_class,
    )

    trainer.train()
    trainer.save_model(output_dir=dpo_config.output_dir)
    processing_class.save_pretrained(dpo_config.output_dir)

    if do_eval:
        ################
        # Evaluation
        ################
        test_metrics = trainer.evaluate()
    else:
        test_metrics = None

    return test_metrics
