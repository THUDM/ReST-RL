from grpo_utils import *
from experiment.args_utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="For policy grpo training")
    parser.add_argument('model', type=str, help='Specify the policy path')
    parser.add_argument('save_dir', type=str, help='Directory path to save checkpoints')
    parser.add_argument('train_pth', type=str, help='Path to the train data')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Specify the checkpoint path to resume training')
    parser.add_argument('--max_prompt_length', default=1024, type=int,
                        help='Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left')
    parser.add_argument('--max_completion_length', default=1024, type=int,
                        help='Maximum length of the generated completion')
    parser.add_argument("--temperature", type=float, default=0.7, help="Specify the temperature for sampling")
    parser.add_argument('--num_generations', type=int, default=8,
                        help='Number of generations per prompt to sample. The global batch size (num_processes * batch_size_per_device) must be divisible by this value')
    parser.add_argument('--deepspeed_config', default=None, type=str, help='Path to the deepspeed config file')
    parser.add_argument('--log_completions', default=False, action='store_true',
                        help='Whether to log the completions during training')
    parser.add_argument('--use_vllm', default=False, action='store_true',
                        help='Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept unused for training, as vLLM will require one for generation')
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9,
                        help="Specify the gpu memory utilization used for vllm generation")
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='Total number of epochs to train the model')
    parser.add_argument('--max_steps', default=-1, type=int, help='If set to a positive number, the total number of training steps to perform')
    parser.add_argument('--save_steps', type=int_or_float, default=0.5, help='Number of training steps(or ratio of total steps) between two saved checkpoints')
    parser.add_argument('--save_only_model', default=False, action='store_true',
                        help='When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--batch_size_per_device', default=4, type=int, help='Batch size for each device')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='Gradient accumulation steps')
    parser.add_argument('--symbol_reward', type=float, default=1e-3, help='Reward for generating appropriate symbols')
    parser.add_argument('--trailing_penalty', type=float, default=1e-6, help='Penalty for generating redundant characters')
    parser.add_argument('--report_to', type=str, default='wandb', help='Way to report train results')
    parser.add_argument('--config_kwargs', nargs='*', action=ParseKwargs, default={})

    args = parser.parse_args()
    return args


def run(args):
    set_reward_items(args)
    train(args)


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments)
