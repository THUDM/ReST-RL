import sys
import os.path
import json

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cur_dir))
import torch
import deepspeed
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoConfig
from verifiers.simple import *
from prompts.stops import get_stop_strings

symbol_reward = 1e-3
trailing_penalty = 1e-6


def set_reward_items(args):
    global symbol_reward, trailing_penalty
    symbol_reward = args.symbol_reward
    trailing_penalty = args.trailing_penalty


# Define the reward function for each domain
def get_reward(prompts: list[str], completions: list[str], init_state: list[str], partial_state: list[str],
               domain: list[str], test: list[str], **kwargs) -> list[float]:
    global symbol_reward, trailing_penalty
    rewards = []
    for i in range(len(prompts)):
        end_strings = get_stop_strings(domain[i])
        if domain[i] == 'DS1000':
            use_extra_reward = True
            try:
                verifier = DS1000Verifier(reference=test[i])
                init_success = True
            except Exception as e:
                print(f"Failed to initialize DS1000 verifier with error: {e}\n")
                verifier = None
                init_success = False

        elif domain[i] == 'APPS':
            use_extra_reward = True
            try:
                verifier = APPSVerifier(reference=json.loads(test[i]))
                init_success = True
            except Exception as e:
                print(f"Failed to initialize APPS verifier with error: {e}\n")
                verifier = None
                init_success = False

        elif domain[i] == 'BigCodeBench':
            use_extra_reward = True
            try:
                verifier = BigCodeBenchVerifier(reference=test[i])
                init_success = True
            except Exception as e:
                print(f"Failed to initialize BigCodeBench verifier with error: {e}\n")
                verifier = None
                init_success = False

        else:
            raise ValueError("Invalid domain")

        # calculate reward
        reward = 0.0
        valid_completion = completions[i]

        if use_extra_reward:
            for end_string in end_strings:
                valid_completion = valid_completion.split(end_string)[0]

            # symbol reward and trailing penalty
            if len(completions[i]) > len(valid_completion):
                reward += symbol_reward
                trailing = completions[i][len(valid_completion):]
                for end_string in end_strings:
                    if trailing.startswith(end_string):
                        trailing = trailing[len(end_string):] if len(trailing) > len(end_string) else ""
                        break
                reward -= trailing_penalty * len(trailing)

        # verification reward
        full_completion = partial_state[i] + valid_completion
        if not init_success:
            reward += 0.0
        else:
            try:
                reward += verifier.verify([full_completion], initial_state=init_state[i])[0]
            except Exception as e:
                print(f"Failed to verify completion with error: {e}\n")
                reward += 0.0

        # append reward
        rewards.append(reward)

    return rewards


def load_deepspeed_config(args):
    if args.deepspeed_config is None:
        return None
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


def set_grpo_training_args(args, model_config):
    deepspeed_config = load_deepspeed_config(args)

    training_args = GRPOConfig(
        max_prompt_length=args.max_prompt_length, max_completion_length=args.max_completion_length,
        use_vllm=args.use_vllm, learning_rate=args.lr, log_completions=args.log_completions,
        output_dir=args.save_dir, per_device_train_batch_size=args.batch_size_per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps, weight_decay=args.weight_decay,
        num_train_epochs=args.epochs, save_strategy="steps", save_steps=args.save_steps, deepspeed=deepspeed_config,
        bf16=True if model_config.torch_dtype == torch.bfloat16 else False,
        fp16=True if model_config.torch_dtype == torch.float16 else False,
        report_to=args.report_to, temperature=args.temperature, max_steps=args.max_steps, save_only_model=args.save_only_model,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization, num_generations=args.num_generations,
        model_init_kwargs={"trust_remote_code": True, "torch_dtype": model_config.torch_dtype},
        **args.config_kwargs,
    )
    return training_args


# Do GRPO training
def train(args):
    model_config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    training_args = set_grpo_training_args(args, model_config)
    train_dataset = load_dataset("json", data_files=args.train_pth, split="train")
    train_dataset = train_dataset.shuffle(seed=42)
    processing_class = AutoTokenizer.from_pretrained(args.model, padding_side="left", trust_remote_code=True)
    print("Using torch dtype: ", model_config.torch_dtype, '\n')

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=get_reward,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processing_class,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output_dir=training_args.output_dir)
    processing_class.save_pretrained(training_args.output_dir)
