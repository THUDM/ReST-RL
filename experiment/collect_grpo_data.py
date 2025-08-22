import argparse
from tqdm import tqdm
from utils.json_operator import *
from experiment.args_utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Collect generated grpo samples for reward distribution analysis")
    # basic settings
    parser.add_argument("--domains", type=str_list, default=["BigCodeBench", "DS1000", "APPS"],
                        help="Specify the datasets")
    parser.add_argument("--backend", type=str, default="gpt-4o-mini", help="Specify the policy backend")
    parser.add_argument("--temperature", type=float, default=0.7, help="Specify the temperature for sampling")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Specify the maximum number of tokens available for generate")

    # sample settings
    parser.add_argument("--completion_accept_threshold_grpo", type=float, default=0.9,
                        help="Specify the reward threshold for accepted completion used for grpo")
    parser.add_argument("--std_accept_threshold_grpo", type=float, default=0.05,
                        help="Specify the reward standard deviation threshold for grpo")
    parser.add_argument("--n_sample", type=int_or_float, default=0.5,
                        help="Specify the number or ratio of sub-samples for each data sample used for grpo. If not positive, only basic grpo data will be extracted")
    parser.add_argument("--extract_all_lines", action="store_true", default=False,
                        help="Specify whether to extract all lines as sub-samples for grpo training")
    parser.add_argument("--alpha", type=float, default=0.95,
                        help="Specify the decay factor used for grpo data sampling")

    args = parser.parse_args()
    return args


def convert_data(domain: str, id: str, prompt: str, code: str, test: str):
    if domain == "BigCodeBench":
        return {
            'task_id': id, 'instruct_prompt': prompt, 'code_prompt': code, 'test': test
        }
    elif domain == "DS1000":
        return {
            'metadata': {'problem_id': id}, 'prompt': prompt, 'code': code, 'code_context': test
        }
    elif domain == "APPS":
        return {
            'id': id, 'prompt': prompt, 'code': code, 'test': test
        }
    else:
        raise ValueError(f"Unknown domain: {domain}")


def collect_grpo_data(args):
    if args.extract_all_lines:
        grpo_filename = f"temp_{args.temperature}_tokens_{args.max_tokens}_grpo_data_{args.std_accept_threshold_grpo}_{args.completion_accept_threshold_grpo}_all.jsonl"
    else:
        if args.n_sample <= 0:
            grpo_filename = f"temp_{args.temperature}_tokens_{args.max_tokens}_grpo_data_basic.jsonl"
        else:
            grpo_filename = f"temp_{args.temperature}_tokens_{args.max_tokens}_grpo_data_{args.std_accept_threshold_grpo}_{args.completion_accept_threshold_grpo}_{args.n_sample}_{args.alpha}.jsonl"

    for domain in args.domains:
        output_source_dir = f"generate/Common/{domain}/{args.backend.replace('/', '--')}"
        out_filename = f"{output_source_dir}/{grpo_filename}"
        output_items = read_json(out_filename)
        if len(output_items) == 0:
            print(f"No output found in <{out_filename}>, skipping...\n")
            continue
        else:
            print(f"Found {len(output_items)} output(s) in <{out_filename}>\n")
            new_prompt_datas = []
            new_prompt_filename = f"{output_source_dir}/{grpo_filename.rstrip('.jsonl')}_prompts.jsonl"
            for item in tqdm(output_items):
                new_prompt_data = convert_data(domain, item['id'], item['prompt'], item['partial_state'], item['test'])
                new_prompt_datas.append(new_prompt_data)

            dump_json(new_prompt_filename, new_prompt_datas)


if __name__ == "__main__":
    args = parse_args()
    collect_grpo_data(args)
