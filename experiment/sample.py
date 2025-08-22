import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable tokenizer parallelism
import argparse
import torch
from tasks.Common.task import *
from prompts.formatters import *
from verifiers.simple import *
from data.reader import *
from experiment.args_utils import *
from utils.json_operator import *
from models.build_llm import build_llm
from prompts.stops import get_stop_strings
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="For common data sampling")
    parser.add_argument("--domain", type=str, default="BigCodeBench",
                        choices=["BigCodeBench", "DS1000", "APPS"],
                        help="Specify the dataset")
    parser.add_argument("--directory", type=str, default=None, help="Specify the data directory")
    parser.add_argument("--file", type=str, default=None, help="Specify the data file")
    parser.add_argument("--idx_list", type=int_list, default=None, help="Specify the data indices")
    parser.add_argument("--backend", type=str, default="gpt-4o-mini", help="Specify the policy backend")
    parser.add_argument("--use_api", action='store_true', default=False, help="Specify whether to use an API")
    parser.add_argument("--low_b", type=float, default=0, help="Specify the low bound of reward")
    parser.add_argument("--n", type=int, default=5,
                        help="Specify the number of samples generated when sampling")
    parser.add_argument("--temperature", type=float, default=0.7, help="Specify the temperature for sampling")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Specify the maximum number of tokens available for generate")
    parser.add_argument("--stop", type=str_list, default=None,
                        help="Specify the stop strings, commas should not be included in any of these strings")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=-1,
                        help="Specify the number of devices used for vllm parallel generation")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9,
                        help="Specify the gpu memory utilization used for vllm generation")
    parser.add_argument("--do_not_format", action='store_true', default=False,
                        help="Specify whether not to format prompts")
    parser.add_argument('--kwargs', nargs='*', action=ParseKwargs, default={})

    args = parser.parse_args()
    return args


def run(args):
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
    if args.domain == "BigCodeBench":
        run_big_code_bench(idx_list=args.idx_list, backend=args.backend, use_api=args.use_api, low_b=args.low_b,
                           n=args.n, temperature=args.temperature, max_tokens=args.max_tokens,
                           stop=args.stop, vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
                           vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                           do_not_format=args.do_not_format,
                           **designated_paths, **args.kwargs)
    elif args.domain == "DS1000":
        run_ds1000(idx_list=args.idx_list, backend=args.backend, use_api=args.use_api, low_b=args.low_b,
                   n=args.n, temperature=args.temperature, max_tokens=args.max_tokens, stop=args.stop,
                   vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
                   vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization, do_not_format=args.do_not_format,
                   **designated_paths, **args.kwargs)
    elif args.domain == "APPS":
        run_apps(idx_list=args.idx_list, backend=args.backend, use_api=args.use_api, low_b=args.low_b,
                 n=args.n, temperature=args.temperature, max_tokens=args.max_tokens, stop=args.stop,
                 vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
                 vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization, do_not_format=args.do_not_format,
                 **designated_paths, **args.kwargs)
    else:
        print("Invalid domain...")


def run_big_code_bench(directory: str = "data/BigCodeBench", file: str = "data.json", idx_list: list[int] = None,
                       backend: str = "gpt-4o-mini", use_api: bool = False, low_b: float = 0, n: int = 5,
                       temperature: float = 0.7, max_tokens: int = 1024, stop: list[str] = None,
                       vllm_tensor_parallel_size: int = -1, vllm_gpu_memory_utilization: float = 0.9,
                       do_not_format: bool = False, **formatter_kwargs):
    # read data
    reader = BigCodeBenchReader(directory=directory, file=file)
    n_data = reader.get_n_data()

    # init llm for vllm engine
    num_cuda_devices = torch.cuda.device_count()
    if not use_api:
        assert num_cuda_devices > 0, "No available CUDA devices detected. Please ensure at least one device is available."

        if vllm_tensor_parallel_size == -1:
            vllm_tensor_parallel_size = num_cuda_devices
        else:
            vllm_tensor_parallel_size = min(vllm_tensor_parallel_size, num_cuda_devices)

        llm = build_llm(backend, tensor_parallel_size=vllm_tensor_parallel_size,
                        gpu_memory_utilization=vllm_gpu_memory_utilization)

    else:
        llm = None

    # init formatter
    tokenizer = AutoTokenizer.from_pretrained(backend, trust_remote_code=True) if not use_api else None
    if tokenizer is None:
        print("Warning: tokenizer is None...\n")
    if tokenizer.chat_template is None:
        print("Warning: tokenizer chat template is None...\n")
    formatter = BigCodeBenchFormatter(domain="BigCodeBench", tokenizer=tokenizer, **formatter_kwargs)

    # set output dirs
    output_dir = f"generate/Common/BigCodeBench/{backend.replace('/', '--')}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not do_not_format:
        out_filename = f"{output_dir}/temp_{temperature}_tokens_{max_tokens}_completions.jsonl"
    else:
        out_filename = f"{output_dir}/temp_{temperature}_tokens_{max_tokens}_completions_no_format.jsonl"
    output_items = read_json(out_filename)

    # set target datas
    if idx_list is None:
        # default: continue from previous stop point
        idx_list = range(len(output_items), n_data)

    # start sampling
    for idx in tqdm(idx_list):
        print(f"Sampling on data <{idx + 1}/{n_data}>")
        data = reader.get_data(idx)
        cur_id = data["id"]
        print(f"Data ID <{cur_id}>")
        q = data['prompt']
        initial_state = data['code']
        test = data['test']
        token_usage = {'prompt': 0, 'completion': 0}
        result = {}

        # init verifier
        try:
            verifier = BigCodeBenchVerifier(domain="BigCodeBench", reference=test, low_b=low_b)
            success = True
        except Exception as e:
            print(f"Failed to initialize verifier with error: {e}\n")
            verifier = None
            success = False

        # init task and run
        if success:
            task = Common_Task(q=q, backend=backend, llm=llm, use_api=use_api, formatter=formatter, verifier=verifier,
                               initial_state=initial_state, phase='train', num_sample=n, temperature=temperature,
                               max_tokens=max_tokens, stop=stop, do_format=not do_not_format)
            result = task.run()
            print('-' * 70, "Results", '-' * 70)
            task.show_token_usage()
            token_usage = task.get_token_usage()

            # completions
            completions = result['completions']
            print(f"Result: {len(completions)} completions found...\n")
            for i, completion in enumerate(completions):
                print('-' * 50, f"Completion {i + 1}", '-' * 50)
                print(
                    f"<Sample>\n{completion['sample']}\n<Formatted Sample>\n{completion['formatted_sample']}\n<Formatted Completion>\n{completion['formatted_completion']}\n<Reward>\n{completion['reward']}\n")

        # output results
        output_item = {'id': cur_id, 'prompt': q, 'init_state': initial_state, 'test': test, 'token_usage': token_usage,
                       'result': result}
        output_items.append(output_item)
        dump_json_line(out_filename, output_item)


def run_ds1000(directory: str = "data/DS1000", file: str = "data.jsonl", idx_list: list[int] = None,
               backend: str = "gpt-4o-mini", use_api: bool = False, low_b: float = 0, n: int = 5,
               temperature: float = 0.7, max_tokens: int = 1024, stop: list[str] = None,
               vllm_tensor_parallel_size: int = -1, vllm_gpu_memory_utilization: float = 0.9,
               do_not_format: bool = False, **formatter_kwargs):
    # read data
    reader = DS1000Reader(directory=directory, file=file)
    n_data = reader.get_n_data()

    # init llm for vllm engine
    num_cuda_devices = torch.cuda.device_count()
    if not use_api:
        assert num_cuda_devices > 0, "No available CUDA devices detected. Please ensure at least one device is available."

        if vllm_tensor_parallel_size == -1:
            vllm_tensor_parallel_size = num_cuda_devices
        else:
            vllm_tensor_parallel_size = min(vllm_tensor_parallel_size, num_cuda_devices)

        llm = build_llm(backend, tensor_parallel_size=vllm_tensor_parallel_size,
                        gpu_memory_utilization=vllm_gpu_memory_utilization)

    else:
        llm = None

    # init formatter
    tokenizer = AutoTokenizer.from_pretrained(backend, trust_remote_code=True) if not use_api else None
    if tokenizer is None:
        print("Warning: tokenizer is None...\n")
    if tokenizer.chat_template is None:
        print("Warning: tokenizer chat template is None...\n")
    formatter = DS1000Formatter(domain="DS1000", tokenizer=tokenizer, **formatter_kwargs)

    # set output dirs
    output_dir = f"generate/Common/DS1000/{backend.replace('/', '--')}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not do_not_format:
        out_filename = f"{output_dir}/temp_{temperature}_tokens_{max_tokens}_completions.jsonl"
    else:
        out_filename = f"{output_dir}/temp_{temperature}_tokens_{max_tokens}_completions_no_format.jsonl"
    output_items = read_json(out_filename)

    # set target datas
    if idx_list is None:
        # default: continue from previous stop point
        idx_list = range(len(output_items), n_data)

    # start sampling
    for idx in tqdm(idx_list):
        print(f"Sampling on data <{idx + 1}/{n_data}>")
        data = reader.get_data(idx)
        cur_id = data["id"]
        print(f"Data ID <{cur_id}>")
        q = data['prompt']
        initial_state = data['code']
        test = data['test']
        token_usage = {'prompt': 0, 'completion': 0}
        result = {}

        # init verifier
        try:
            verifier = DS1000Verifier(domain="DS1000", reference=test, low_b=low_b)
            success = True
        except Exception as e:
            print(f"Failed to initialize verifier with error: {e}\n")
            verifier = None
            success = False

        # init task and run
        if success:
            task = Common_Task(q=q, backend=backend, llm=llm, use_api=use_api, formatter=formatter, verifier=verifier,
                               initial_state=initial_state, phase='train', num_sample=n, temperature=temperature,
                               max_tokens=max_tokens, stop=stop, do_format=not do_not_format)
            result = task.run()
            print('-' * 70, "Results", '-' * 70)
            task.show_token_usage()
            token_usage = task.get_token_usage()

            # completions
            completions = result['completions']
            print(f"Result: {len(completions)} completions found...\n")
            for i, completion in enumerate(completions):
                print('-' * 50, f"Completion {i + 1}", '-' * 50)
                print(
                    f"<Sample>\n{completion['sample']}\n<Formatted Sample>\n{completion['formatted_sample']}\n<Formatted Completion>\n{completion['formatted_completion']}\n<Reward>\n{completion['reward']}\n")

        # output results
        output_item = {'id': cur_id, 'prompt': q, 'init_state': initial_state, 'test': test, 'token_usage': token_usage,
                       'result': result}
        output_items.append(output_item)
        dump_json_line(out_filename, output_item)


def run_apps(directory: str = "data/APPS", file: str = "data_with_test.jsonl", idx_list: list[int] = None,
             backend: str = "gpt-4o-mini", use_api: bool = False, low_b: float = 0, n: int = 5,
             temperature: float = 0.7, max_tokens: int = 1024, stop: list[str] = None,
             vllm_tensor_parallel_size: int = -1, vllm_gpu_memory_utilization: float = 0.9,
             do_not_format: bool = False, **formatter_kwargs):
    # read data
    reader = APPSReader(directory=directory, file=file)
    n_data = reader.get_n_data()

    # init llm for vllm engine
    num_cuda_devices = torch.cuda.device_count()
    if not use_api:
        assert num_cuda_devices > 0, "No available CUDA devices detected. Please ensure at least one device is available."

        if vllm_tensor_parallel_size == -1:
            vllm_tensor_parallel_size = num_cuda_devices
        else:
            vllm_tensor_parallel_size = min(vllm_tensor_parallel_size, num_cuda_devices)

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

    # set output dirs
    output_dir = f"generate/Common/APPS/{backend.replace('/', '--')}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not do_not_format:
        out_filename = f"{output_dir}/temp_{temperature}_tokens_{max_tokens}_completions.jsonl"
    else:
        out_filename = f"{output_dir}/temp_{temperature}_tokens_{max_tokens}_completions_no_format.jsonl"
    output_items = read_json(out_filename)

    # set target datas
    if idx_list is None:
        # default: continue from previous stop point
        idx_list = range(len(output_items), n_data)

    # start sampling
    for idx in tqdm(idx_list):
        print(f"Sampling on data <{idx + 1}/{n_data}>")
        data, success = reader.get_data(idx)
        cur_id = data["id"]
        print(f"Data ID <{cur_id}>")
        q = data['prompt']
        initial_state = data['code']
        test = data['test']
        token_usage = {'prompt': 0, 'completion': 0}
        result = {}

        # init verifier
        if success:
            try:
                verifier = APPSVerifier(domain="APPS", reference=test, low_b=low_b)
            except Exception as e:
                print(f"Failed to initialize verifier with error: {e}\n")
                verifier = None
                success = False
        else:
            print("Already failed to read data, skip verifier initialization\n")
            verifier = None

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
            print("Already failed to read data or initialize verifier, skip formatter initialization\n")
            formatter = None

        # init task and run
        if success:
            task = Common_Task(q=q, backend=backend, llm=llm, use_api=use_api, formatter=formatter, verifier=verifier,
                               initial_state=initial_state, phase='train', num_sample=n, temperature=temperature,
                               max_tokens=max_tokens, stop=stop, do_format=not do_not_format)
            result = task.run()
            print('-' * 70, "Results", '-' * 70)
            task.show_token_usage()
            token_usage = task.get_token_usage()

            # completions
            completions = result['completions']
            print(f"Result: {len(completions)} completions found...\n")
            for i, completion in enumerate(completions):
                print('-' * 50, f"Completion {i + 1}", '-' * 50)
                print(
                    f"<Sample>\n{completion['sample']}\n<Formatted Sample>\n{completion['formatted_sample']}\n<Formatted Completion>\n{completion['formatted_completion']}\n<Reward>\n{completion['reward']}\n")

        # output results
        output_item = {'id': cur_id, 'prompt': q, 'init_state': initial_state, 'test': test, 'token_usage': token_usage,
                       'result': result}
        output_items.append(output_item)
        dump_json_line(out_filename, output_item)


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments)
