import sys
import argparse
import os.path
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cur_dir))
from rms.reward_models import *
from experiment.args_utils import int_or_float


def parse_args():
    parser = argparse.ArgumentParser(description="For reward model training")
    parser.add_argument('backend', type=str, help='Specify the backend')
    parser.add_argument('save_dir', type=str, help='Directory path to save checkpoints')
    parser.add_argument('train_pth', type=str, help='Path to the train data')
    parser.add_argument('rm_class', type=str, choices=['standard_scalar', 'transformers_scalar', 'transformers_prob'], help='Specify the reward model class')
    parser.add_argument('--test_pth', type=str, default=None, help='Path to the test data, if exists')
    parser.add_argument('--state_pth', type=str, default=None, help='Path to the previous state dict, if exists')
    parser.add_argument('--rm_type', type=str, default='prm', choices=['prm', 'orm'], help='Specify the reward model type')
    parser.add_argument('--save_every', type=int, default=1, help='How often (by epoch) to save a checkpoint')
    parser.add_argument('--max_steps', default=-1, type=int,
                        help='If set to a positive number, the total number of training steps to perform')
    parser.add_argument('--save_steps', type=int_or_float, default=0.25,
                        help='Number of training steps(or ratio of total steps) between two saved checkpoints')
    parser.add_argument('--save_only_model', default=False, action='store_true',
                        help='When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state')
    parser.add_argument('--max_length', default=1024, type=int, help='Max length of the input sequence')
    parser.add_argument('--paradigm', default='transformers_deepspeed', choices=['common', 'pytorch_ddp', 'transformers_deepspeed'], help='Use paradigms like pytorch ddp for training')
    parser.add_argument('--test_tol', type=float, default=0.05, help='Tolerance for test')
    parser.add_argument('--loss_weight_factor', type=float, default=0.5, help='Weight factor used for cross entropy loss compute, only effective when adopting a transformers_prob class reward model')
    parser.add_argument('--deepspeed_config', default='rms/config/zero3_config.json', type=str,
                        help='Path to the deepspeed config file')
    parser.add_argument('--lr', type=float, default=1e-7, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='Total epochs to train the model')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--batch_size_per_device', default=4, type=int, help='Batch size for each device')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='Gradient accumulation steps')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed from distributed launcher')
    parser.add_argument('--min_rw', type=float, default=0.0, help='Minimum reward')
    parser.add_argument('--max_rw', type=float, default=1.0, help='Maximum reward')
    parser.add_argument('--report_to', type=str, default='wandb', help='Way to report train results')

    args = parser.parse_args()
    return args


def run(args):
    # prepare
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    paradigm = args.paradigm
    if args.rm_class == 'standard_scalar':
        model = StandardScalarRM(args.backend, args.rm_type, device, args.state_pth, args.max_length, args.min_rw, args.max_rw)
        assert paradigm in ['common', 'pytorch_ddp'], 'Only common and pytorch_ddp training paradigms are supported for standard scalar reward models\n'
    elif args.rm_class == 'transformers_scalar':
        model = TransformersScalarRM(args.backend, args.rm_type, 'train', args.max_length, args.min_rw, args.max_rw)
        assert paradigm == 'transformers_deepspeed', 'Only transformers_deepspeed training paradigm is supported for transformers scalar reward models\n'
    else:
        model = TransformersProbRM(args.backend, args.rm_type, 'train', args.max_length)
        assert paradigm == 'transformers_deepspeed', 'Only transformers_deepspeed training paradigm is supported for transformers probability reward models\n'

    # train
    if paradigm == 'transformers_deepspeed':
        result = model.train(args)

        # save results
        save_result_path = os.path.join(args.save_dir, "test_results.jsonl")
        dump_json(save_result_path, [result])

    else:
        if paradigm == 'pytorch_ddp':
            results = model.ddp_train(args.train_pth, args.test_pth, args.batch_size_per_device, args.lr, args.epochs, args.save_dir, args.save_every, args.test_tol)
        else:
            results = model.train(args.train_pth, args.test_pth, args.batch_size_per_device, args.lr, args.epochs, args.save_dir, args.save_every, args.test_tol)

        # save results
        best = max(results, key=lambda x: x['result']['pass_rate'])
        print(
            f"Best model: <{best['model']}>\nPassed: {best['result']['passed']} | Total: {best['result']['total']} | Pass rate: {best['result']['pass_rate']}\n")
        save_result_path = os.path.join(args.save_dir, "test_results.jsonl")
        dump_json(save_result_path, results)


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments)
