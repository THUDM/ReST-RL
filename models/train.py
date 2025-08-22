import sys
import argparse
import os.path
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cur_dir))
from models.dpo_utils import dpo_train
from models.sft_utils import sft_train
from utils.json_operator import *
from experiment.args_utils import int_or_float


def parse_args():
    parser = argparse.ArgumentParser(description="For policy training")
    parser.add_argument('model', type=str, help='Specify the policy path')
    parser.add_argument('save_dir', type=str, help='Directory path to save checkpoints')
    parser.add_argument('train_pth', type=str, help='Path to the train data')
    parser.add_argument('--test_pth', type=str, default=None, help='Path to the test data, if exists')
    parser.add_argument('--mode', type=str, choices=['sft', 'dpo'], default='sft', help='Specify the training mode')
    parser.add_argument('--max_seq_length', default=2048, type=int, help='Max length of the full input sequence')
    parser.add_argument('--max_prompt_length', default=1024, type=int,
                        help='Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left')
    parser.add_argument('--max_completion_length', default=1024, type=int,
                        help='Maximum length of the completion')
    parser.add_argument('--deepspeed_config', default=None, type=str,
                        help='Path to the deepspeed config file')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='Total epochs to train the model')
    parser.add_argument('--max_steps', default=-1, type=int,
                        help='If set to a positive number, the total number of training steps to perform')
    parser.add_argument('--save_steps', type=int_or_float, default=0.5,
                        help='Number of training steps(or ratio of total steps) between two saved checkpoints')
    parser.add_argument('--save_only_model', default=False, action='store_true',
                        help='When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--batch_size_per_device', default=4, type=int, help='Batch size for each device')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='Gradient accumulation steps')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed from distributed launcher')
    parser.add_argument('--report_to', type=str, default='wandb', help='Way to report train results')

    args = parser.parse_args()
    return args


def run(args):
    # prepare
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # train
    mode = args.mode
    if mode == 'sft':
        result = sft_train(args)

        # save results
        if result is not None:
            save_result_path = os.path.join(args.save_dir, "test_results.jsonl")
            dump_json(save_result_path, [result])

    elif mode == 'dpo':
        result = dpo_train(args)

        # save results
        if result is not None:
            save_result_path = os.path.join(args.save_dir, "test_results.jsonl")
            dump_json(save_result_path, [result])

    else:
        print("Invalid training mode\n")


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments)
