import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable tokenizer parallelism
import argparse
import torch
from transformers import AutoTokenizer
from tasks.Common.task import Common_Task
from tasks.MCTS.task import MCTS_Task
from prompts.formatters import *
from verifiers.simple import *
from rms.deploy_utils import deploy_rm
from data.reader import *
from tasks.MCTS.tree_utils import *
from experiment.args_utils import *
from utils.json_operator import *
from models.build_llm import build_llm
from prompts.stops import get_stop_strings
from evaluation.utils import progress
from termcolor import cprint


def parse_args():
    parser = argparse.ArgumentParser(description="For evaluation")
    # basic settings
    parser.add_argument("--method", type=str, default="Common", choices=["Common", "MCTS"],
                        help="Specify the evaluation method")
    parser.add_argument("--domain", type=str, default="APPS", choices=["APPS"],
                        help="Specify the dataset")
    parser.add_argument("--directory", type=str, default=None, help="Specify the data directory")
    parser.add_argument("--file", type=str, default=None, help="Specify the data file")
    parser.add_argument("--idx_list", type=int_list, default=None, help="Specify the data indices")
    parser.add_argument("--backend", type=str, default="gpt-4o-mini", help="Specify the policy backend")
    parser.add_argument("--use_api", action='store_true', default=False, help="Specify whether to use an API")

    # rm settings
    parser.add_argument("--rm_backend", type=str, default=None, help="Specify the reward model backend")
    parser.add_argument("--rm_class", type=str, default="transformers_scalar",
                        choices=["transformers_scalar", "transformers_prob", "skywork", "intern", "generalizable",
                                 "qwen", "rlhflow"],
                        help="Specify the reward model class")
    parser.add_argument("--rm_type", type=str, default="prm", choices=["prm", "orm"],
                        help="Specify the reward model type")
    parser.add_argument("--max_length", type=int, default=2048, help="Specify the max length for reward model")

    # mcts settings
    parser.add_argument("--time_limit", type=float, default=None, help="Specify the time limit")
    parser.add_argument("--iteration_limit", type=int, default=15, help="Specify the iteration limit")
    parser.add_argument("--num_decision", type=int, default=3,
                        help="Specify the number of samples retained when sampling in mcts")
    parser.add_argument("--exploration_constant", type=float, default=0.1,
                        help="Specify the exploration constant for UCT")
    parser.add_argument("--eps", type=float, default=0.1, help="Specify the constant used for handling zero division")

    # other settings
    parser.add_argument("--num_sample", type=int, default=5,
                        help="If using mcts, the number of samples generated for each iteration. Else, the total number of samples to be evaluated by the reward model")
    parser.add_argument("--n", type=int, default=1, help="Specify the number of samples generated for evaluation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Specify the temperature for sampling")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Specify the maximum number of tokens available for generate")
    parser.add_argument("--stop", type=str_list, default=None,
                        help="Specify the stop strings, commas should not be included in any of these strings")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=-1,
                        help="Specify the number of devices used for vllm parallel generation")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9,
                        help="Specify the gpu memory utilization used for vllm generation")
    parser.add_argument('--kwargs', nargs='*', action=ParseKwargs, default={})

    args = parser.parse_args()
    return args


def evaluate(args):
    # specified paths
    designated_paths = {}
    if args.directory is not None:
        designated_paths["directory"] = args.directory
    if args.file is not None:
        designated_paths["file"] = args.file

    # stop strings
    if args.stop is None:
        args.stop = get_stop_strings(args.domain)

    # select dateset
    if args.domain == "APPS":
        eval_apps(method=args.method, idx_list=args.idx_list, backend=args.backend, use_api=args.use_api,
                  rm_backend=args.rm_backend, rm_class=args.rm_class, rm_type=args.rm_type, max_length=args.max_length,
                  time_limit=args.time_limit, iteration_limit=args.iteration_limit,
                  num_sample=args.num_sample, num_decision=args.num_decision,
                  exploration_constant=args.exploration_constant, eps=args.eps, n=args.n, temperature=args.temperature,
                  max_tokens=args.max_tokens, stop=args.stop,
                  vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
                  vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                  **designated_paths, **args.kwargs)
    else:
        print("Invalid domain...")


def eval_apps(directory: str = "data/APPS", file: str = "test_500.jsonl", method: str = "Common",
              idx_list: list[int] = None,
              backend: str = "gpt-4o-mini", use_api: bool = False, rm_backend: str = None,
              rm_class: str = "transformers_scalar",
              rm_type: str = "prm", max_length: int = 2048, time_limit: float = None, iteration_limit: int = None,
              num_sample: int = 5,
              num_decision: int = 3, exploration_constant: float = 0.1, eps: float = 0.1, n: int = 1,
              temperature: float = 0.7, max_tokens: int = 1024, stop: list[str] = None,
              vllm_tensor_parallel_size: int = -1, vllm_gpu_memory_utilization: float = 0.9, **formatter_kwargs):
    # read data
    reader = APPSReader(directory=directory, file=file)
    n_data = reader.get_n_data()

    # init llm for vllm engine, automatically detect available cuda devices for the engine
    num_cuda_devices = torch.cuda.device_count()
    if not use_api:
        assert num_cuda_devices > 0, "No available CUDA devices detected. Please ensure at least one device is available."

        if vllm_tensor_parallel_size == -1:
            if method == 'Common':
                if rm_backend is None:
                    vllm_tensor_parallel_size = num_cuda_devices
                else:
                    assert num_cuda_devices > 1, "For common test with reward model, at least two CUDA devices are required."
                    vllm_tensor_parallel_size = num_cuda_devices - 1
            else:
                assert num_cuda_devices > 1, "For mcts test, at least two CUDA devices are required."
                vllm_tensor_parallel_size = num_cuda_devices - 1
        else:
            if method == 'Common':
                if rm_backend is None:
                    vllm_tensor_parallel_size = min(vllm_tensor_parallel_size, num_cuda_devices)
                else:
                    assert num_cuda_devices > 1, "For common test with reward model, at least two CUDA devices are required."
                    vllm_tensor_parallel_size = min(num_cuda_devices - 1, vllm_tensor_parallel_size)
            else:
                assert num_cuda_devices > 1, "For mcts test, at least two CUDA devices are required."
                vllm_tensor_parallel_size = min(num_cuda_devices - 1, vllm_tensor_parallel_size)

        print(
            f"Detected {num_cuda_devices} CUDA devices. Setting vllm_tensor_parallel_size to {vllm_tensor_parallel_size}.")
        for i in range(vllm_tensor_parallel_size):
            vllm_device = torch.device(f"cuda:{i}")
            print(f"Vllm Device {i}: {vllm_device}")
        llm = build_llm(backend, tensor_parallel_size=vllm_tensor_parallel_size,
                        gpu_memory_utilization=vllm_gpu_memory_utilization)

    else:
        llm = None

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(backend, trust_remote_code=True) if not use_api else None
    if tokenizer is None:
        print("Warning: tokenizer is None...\n")
    if tokenizer.chat_template is None:
        print("Warning: tokenizer chat template is None...\n")

    # init rm
    if method == 'MCTS':
        assert rm_backend is not None, "Reward model is required for mcts test."
        assert num_cuda_devices > 0, "No available CUDA devices detected. Please ensure at least one device is available."
        rm_device_index = num_cuda_devices - 1
        rm_device = torch.device(f"cuda:{rm_device_index}")
        print(f"Reward model will use device: {rm_device}")
        rm = deploy_rm(rm_class=rm_class, backend=rm_backend, rm_type=rm_type, max_length=max_length, device=rm_device)

    else:
        if rm_backend is not None:
            assert num_cuda_devices > 0, "No available CUDA devices detected. Please ensure at least one device is available."
            rm_device_index = num_cuda_devices - 1
            rm_device = torch.device(f"cuda:{rm_device_index}")
            print(f"Reward model will use device: {rm_device}")
            rm = deploy_rm(rm_class=rm_class, backend=rm_backend, rm_type=rm_type, max_length=max_length,
                           device=rm_device)
        else:
            rm = None

    # set output dirs
    output_dir = f"output/apps_results/{method}/{backend.replace('/', '--')}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if method == 'MCTS':
        out_filename = f"{output_dir}/rm_{rm_backend.replace('/', '--')}_time_{time_limit}_iter_{iteration_limit}_sample_{num_sample}_decision_{num_decision}_exp_{exploration_constant}_eps_{eps}_temp_{temperature}_tokens_{max_tokens}_n_{n}_completions.jsonl"
    else:
        if rm_backend is not None:
            out_filename = f"{output_dir}/rm_{rm_backend.replace('/', '--')}_sample_{num_sample}_temp_{temperature}_tokens_{max_tokens}_n_{n}_completions.jsonl"
        else:
            out_filename = f"{output_dir}/temp_{temperature}_tokens_{max_tokens}_n_{n}_completions.jsonl"
    output_items = read_json(out_filename)

    # set target datas
    if idx_list is None:
        # default: continue from previous stop point
        idx_list = range(len(output_items), n_data)

    # start test
    print("Starting test...\n")
    with progress("apps") as p:
        for idx in p.track(idx_list):
            print(f"Testing on data <{idx + 1}/{n_data}>")
            data, success = reader.get_data(idx)
            cur_id = data["id"]
            log = f"Codegen: {cur_id} @ {backend}"
            p.console.print(log)
            q = data['prompt']
            initial_state = data['code']
            test = data['test']

            # init verifier
            verifier = Verifier(domain="APPS")

            # init formatter
            if success:
                try:
                    use_call_style = True if test.get("fn_name") else False
                    formatter = APPSFormatter(domain="APPS", tokenizer=tokenizer, use_call_style=use_call_style,
                                              **formatter_kwargs)
                except Exception as e:
                    print(f"Failed to initialize formatter with error: {e}\n")
                    formatter = None
                    success = False
            else:
                print("Already failed to read data, skip formatter initialization\n")
                formatter = None

            # init task and run
            token_usage = {'prompt': 0, 'completion': 0}
            sample_completions = []
            if success:
                if method == 'MCTS':
                    for _ in range(n):
                        task = MCTS_Task(q=q, backend=backend, llm=llm, use_api=use_api, formatter=formatter,
                                         verifier=verifier,
                                         initial_state=initial_state, phase='test', time_limit=time_limit,
                                         iteration_limit=iteration_limit, num_sample=num_sample,
                                         num_decision=num_decision,
                                         exploration_constant=exploration_constant, eps=eps, temperature=temperature,
                                         max_tokens=max_tokens, stop=stop, rm=rm)
                        tree = task.run()

                        # usage
                        sample_usage = task.get_token_usage()
                        token_usage['prompt'] += sample_usage['prompt']
                        token_usage['completion'] += sample_usage['completion']

                        # completions
                        completion_items = []
                        completions = get_all_completions(tree)
                        if not completions:
                            sample_completions.append("")
                        else:
                            for completion in completions:
                                for key, value in completion.items():
                                    completion_items.append({'completion': key, 'reward': value})

                            # sort completions by reward and output
                            completion_items_sorted = sorted(completion_items, key=lambda x: x['reward'], reverse=True)
                            best_completion = completion_items_sorted[0]['completion']
                            sample_completions.append(best_completion)
                else:
                    if rm_backend is not None:
                        for _ in range(n):
                            task = Common_Task(q=q, backend=backend, llm=llm, use_api=use_api, formatter=formatter,
                                               verifier=verifier,
                                               initial_state=initial_state, phase='test', num_sample=num_sample,
                                               temperature=temperature,
                                               max_tokens=max_tokens, stop=stop, rm=rm)
                            result = task.run()

                            # usage
                            sample_usage = task.get_token_usage()
                            token_usage['prompt'] += sample_usage['prompt']
                            token_usage['completion'] += sample_usage['completion']

                            # completions
                            completions = result['completions']

                            # sort completions by reward and output
                            completions_sorted = sorted(completions, key=lambda x: x['reward'], reverse=True)
                            best_completion = completions_sorted[0]['formatted_completion']
                            sample_completions.append(best_completion)
                    else:
                        task = Common_Task(q=q, backend=backend, llm=llm, use_api=use_api, formatter=formatter,
                                           verifier=verifier,
                                           initial_state=initial_state, phase='test', num_sample=n,
                                           temperature=temperature,
                                           max_tokens=max_tokens, stop=stop)
                        result = task.run()

                        # usage
                        sample_usage = task.get_token_usage()
                        token_usage['prompt'] += sample_usage['prompt']
                        token_usage['completion'] += sample_usage['completion']

                        # completions
                        completions = result['completions']
                        for completion in completions:
                            sample_completions.append(completion['formatted_completion'])

            # output completion results
            output_item = {'id': cur_id, 'init_succeeded': success, 'completions': sample_completions,
                           'token_usage': token_usage}
            output_items.append(output_item)
            dump_json_line(out_filename, output_item)

    # verify
    print("Starting to verify results...\n")
    assert len(output_items) == n_data, "Number of output items does not match the number of data."
    if method == 'MCTS':
        verify_result_filename = f"{output_dir}/rm_{rm_backend.replace('/', '--')}_time_{time_limit}_iter_{iteration_limit}_sample_{num_sample}_decision_{num_decision}_exp_{exploration_constant}_eps_{eps}_temp_{temperature}_tokens_{max_tokens}_n_{n}_verified.jsonl"
    else:
        if rm_backend is not None:
            verify_result_filename = f"{output_dir}/rm_{rm_backend.replace('/', '--')}_sample_{num_sample}_temp_{temperature}_tokens_{max_tokens}_n_{n}_verified.jsonl"
        else:
            verify_result_filename = f"{output_dir}/temp_{temperature}_tokens_{max_tokens}_n_{n}_verified.jsonl"
    eval_items = read_json(verify_result_filename)

    for idx in tqdm(range(len(eval_items), n_data)):
        output_item = output_items[idx]
        data, success = reader.get_data(idx)
        cur_id = data["id"]
        initial_state = data['code']
        test = data['test']

        # init verifier
        if success:
            try:
                verifier = APPSVerifier(domain="APPS", reference=test)
            except Exception as e:
                print(f"Failed to initialize verifier with error: {e}\n")
                verifier = None
                success = False
        else:
            print("Already failed to read data, skip verifier initialization\n")
            verifier = None

        if not output_item['init_succeeded'] or not success:
            eval_item = {'id': cur_id, 'init_succeeded': False, 'rewards': []}
            eval_items.append(eval_item)
            dump_json_line(verify_result_filename, eval_item)
            continue

        # verify
        completions = output_item['completions']
        rewards = verifier.verify(completions, initial_state)
        eval_item = {'id': cur_id, 'init_succeeded': True, 'rewards': rewards}
        eval_items.append(eval_item)
        dump_json_line(verify_result_filename, eval_item)

    # evaluate metrics
    print("Starting to evaluate metrics...\n")
    assert len(eval_items) == n_data, "Number of verified items does not match the number of data."
    if method == 'MCTS':
        eval_result_filename = f"{output_dir}/rm_{rm_backend.replace('/', '--')}_time_{time_limit}_iter_{iteration_limit}_sample_{num_sample}_decision_{num_decision}_exp_{exploration_constant}_eps_{eps}_temp_{temperature}_tokens_{max_tokens}_n_{n}_results.jsonl"
    else:
        if rm_backend is not None:
            eval_result_filename = f"{output_dir}/rm_{rm_backend.replace('/', '--')}_sample_{num_sample}_temp_{temperature}_tokens_{max_tokens}_n_{n}_results.jsonl"
        else:
            eval_result_filename = f"{output_dir}/temp_{temperature}_tokens_{max_tokens}_n_{n}_results.jsonl"

    average_scores = []
    strict_scores = []
    n_succeeded = 0
    n_failed = 0
    for eval_item in tqdm(eval_items):
        if eval_item['init_succeeded']:
            n_succeeded += 1
            rewards = eval_item['rewards']
            average_scores.append(sum(rewards) * 1.0 / len(rewards))
            strict_rewards = [1 if reward > 0.999 else 0 for reward in rewards]
            strict_scores.append(sum(strict_rewards) * 1.0 / len(strict_rewards))
        else:
            n_failed += 1

    average_acc = sum(average_scores) * 1.0 / len(average_scores)
    strict_acc = sum(strict_scores) * 1.0 / len(strict_scores)
    cprint(f"apps ({file})", "green")
    cprint(f"method -- {method}", "green")
    cprint(f"model -- {backend}", "green")
    cprint(
        f"\n--Test Results--\nInit Succeeded: {n_succeeded}, Init Failed: {n_failed}\nAverage Accuracy: {average_acc}, Strict Accuracy: {strict_acc}\n",
        "green")
    eval_result = [{'n_init_succeeded': n_succeeded, 'n_init_failed': n_failed, 'average_acc': average_acc,
                    'strict_acc': strict_acc}]
    dump_json(eval_result_filename, eval_result)


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
