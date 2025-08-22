import argparse
import math
from tqdm import tqdm
import numpy as np
from utils.json_operator import *
from utils.sample_utils import sample_with_exponential_distribution
from experiment.args_utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Process generated samples for further training")
    # basic settings
    parser.add_argument("--domain", type=str, default="code", choices=["code"], help="Specify the task domain")
    parser.add_argument("--backend", type=str, default="gpt-4o-mini", help="Specify the policy backend")
    parser.add_argument("--method", type=str, default="Common", choices=["Common", "MCTS"],
                        help="Specify the sampling method")
    parser.add_argument("--mode", type=str, default="grpo", choices=["grpo", "dpo", "sft"],
                        help="Specify the training mode that will be used for common data")
    parser.add_argument("--temperature", type=float, default=0.7, help="Specify the temperature for sampling")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Specify the maximum number of tokens available for generate")

    # mcts settings
    parser.add_argument("--time_limit", type=float, default=None, help="Specify the time limit")
    parser.add_argument("--iteration_limit", type=int, default=100, help="Specify the iteration limit")
    parser.add_argument("--num_sample", type=int, default=5,
                        help="Specify the number of samples generated when sampling in mcts")
    parser.add_argument("--num_decision", type=int, default=3,
                        help="Specify the number of samples retained when sampling in mcts")
    parser.add_argument("--exploration_constant", type=float, default=0.2,
                        help="Specify the exploration constant for UCT")
    parser.add_argument("--eps", type=float, default=0.1, help="Specify the constant used for handling zero division")
    parser.add_argument("--use_thought", action="store_true", default=False,
                        help="Specify whether to use thought tokens")

    # sample settings
    parser.add_argument("--completion_accept_threshold_sft", type=float, default=0.9,
                        help="Specify the reward threshold for accepted completion used for sft")
    parser.add_argument("--completion_accept_threshold_grpo", type=float, default=0.9,
                        help="Specify the reward threshold for accepted completion used for grpo")
    parser.add_argument("--std_accept_threshold_grpo", type=float, default=0.05,
                        help="Specify the reward standard deviation threshold for grpo")
    parser.add_argument("--do_aggregate", action="store_true", default=False,
                        help="Specify whether to aggregate all the data into one train file")
    parser.add_argument("--n_sample", type=int_or_float, default=0.5,
                        help="Specify the number or ratio of sub-samples for each data sample used for grpo. If not positive, only basic grpo data will be extracted")
    parser.add_argument("--extract_all_lines", action="store_true", default=False,
                        help="Specify whether to extract all lines as sub-samples for grpo training")
    parser.add_argument("--alpha", type=float, default=0.95,
                        help="Specify the decay factor used for grpo data sampling")

    args = parser.parse_args()
    return args


def process_sample(formatted_prompt: str, formatted_sample: str, domain: str):
    if domain == "BigCodeBench":
        processed_sample = f"{formatted_sample}```"
        return processed_sample

    elif domain == "APPS":
        processed_sample = f"{formatted_sample}```"
        return processed_sample

    elif domain == "DS1000":
        if formatted_prompt.rstrip().endswith('<code>'):
            processed_sample = f"{formatted_sample}</code>"
        elif formatted_prompt.rstrip().endswith('### BEGIN SOLUTION'):
            processed_sample = f"{formatted_sample}### END SOLUTION"
        elif formatted_prompt.rstrip().endswith('# SOLUTION START'):
            processed_sample = f"{formatted_sample}# SOLUTION END"
        else:
            processed_sample = formatted_sample
        return processed_sample

    else:
        raise NotImplementedError


def process(args):
    # domain
    if args.domain == "code":
        domains = ["BigCodeBench", "DS1000", "APPS"]
    else:
        raise NotImplementedError

    if args.method == "Common":
        if args.mode == "grpo":
            all_partial_completion_datas = []
            if args.extract_all_lines:
                grpo_filename = f"temp_{args.temperature}_tokens_{args.max_tokens}_grpo_data_{args.std_accept_threshold_grpo}_{args.completion_accept_threshold_grpo}_all.jsonl"
                mode = "all"
            else:
                if args.n_sample <= 0:
                    grpo_filename = f"temp_{args.temperature}_tokens_{args.max_tokens}_grpo_data_basic.jsonl"
                    mode = "basic"
                else:
                    if isinstance(args.n_sample, int):
                        mode = "sample_fixed"
                    else:
                        assert 0 < args.n_sample <= 1, "Invalid n_sample value, float value must be smaller than 1."
                        mode = "sample_ratio"
                    grpo_filename = f"temp_{args.temperature}_tokens_{args.max_tokens}_grpo_data_{args.std_accept_threshold_grpo}_{args.completion_accept_threshold_grpo}_{args.n_sample}_{args.alpha}.jsonl"

            for domain in domains:
                output_source_dir = f"generate/Common/{domain}/{args.backend.replace('/', '--')}"
                out_filename = f"{output_source_dir}/temp_{args.temperature}_tokens_{args.max_tokens}_completions.jsonl"
                output_items = read_json(out_filename)
                if len(output_items) == 0:
                    print(f"No output found in <{out_filename}>, skipping...\n")
                    continue

                else:
                    print(f"Found {len(output_items)} output(s) in <{out_filename}>\n")
                    partial_completion_datas = []
                    for item in tqdm(output_items):
                        # get formatted prompt and completion
                        result = item['result']
                        if not result:
                            print(f"No result found for current item in <{out_filename}>, skipping...\n")
                            continue
                        formatted_prompt = result['formatted_prompt']
                        base_partial_completion_data = {
                            'prompt': formatted_prompt, 'init_state': item['init_state'],
                            'partial_state': item['init_state'], 'domain': domain,
                            'test': json.dumps(item['test']) if isinstance(item['test'], dict) else item['test'],
                            'id': str(item['id'])
                        }

                        if mode == "basic":
                            # basic grpo
                            partial_completion_datas.append(base_partial_completion_data)
                            continue

                        # get completions, rewards and calculate std
                        completions = result['completions']
                        rewards = [x['reward'] for x in completions]
                        std = np.std(rewards)
                        if std < args.std_accept_threshold_grpo:
                            # skip since std is small
                            continue
                        partial_completion_datas.append(base_partial_completion_data)
                        completion_with_max_reward_item = max(completions, key=lambda x: x['reward'])
                        completion = completion_with_max_reward_item['formatted_sample'].rstrip()
                        reward = completion_with_max_reward_item['reward']
                        if reward < args.completion_accept_threshold_grpo:
                            # skip sub-samples since reward is small
                            continue

                        # extract sub-samples
                        sample_partial_completion_datas = []
                        completion_lines = completion.split('\n')
                        partial_completion = ""
                        for line in completion_lines:
                            partial_completion += line + '\n'
                            sample_partial_completion_datas.append(
                                {'prompt': formatted_prompt + partial_completion, 'init_state': item['init_state'],
                                 'partial_state': item['init_state'] + partial_completion, 'domain': domain,
                                 'test': json.dumps(item['test']) if isinstance(item['test'], dict) else item['test'],
                                 'id': str(item['id'])})

                        # sample with exponential distribution or extract all lines
                        if mode == "all":
                            partial_completion_datas.extend(sample_partial_completion_datas)
                        else:
                            sample_num = args.n_sample if mode == "sample_fixed" else math.ceil(len(sample_partial_completion_datas) * args.n_sample)
                            partial_completion_datas.extend(
                                sample_with_exponential_distribution(sample_partial_completion_datas, args.alpha, sample_num)
                            )

                    # write domain data
                    partial_completions_filename = f"{output_source_dir}/{grpo_filename}"
                    dump_json(partial_completions_filename, partial_completion_datas)
                    all_partial_completion_datas.extend(partial_completion_datas)

            # write all data
            if args.do_aggregate:
                all_output_dir = f"generate/Common/All/{args.backend.replace('/', '--')}"
                if not os.path.exists(all_output_dir):
                    os.makedirs(all_output_dir)
                all_partial_completions_filename = f"{all_output_dir}/{grpo_filename}"
                dump_json(all_partial_completions_filename, all_partial_completion_datas)

        elif args.mode == "dpo":
            all_preference_datas = []
            dpo_filename = f"temp_{args.temperature}_tokens_{args.max_tokens}_dpo_data.jsonl"

            for domain in domains:
                output_source_dir = f"generate/Common/{domain}/{args.backend.replace('/', '--')}"
                out_filename = f"{output_source_dir}/temp_{args.temperature}_tokens_{args.max_tokens}_completions.jsonl"
                output_items = read_json(out_filename)
                if len(output_items) == 0:
                    print(f"No output found in <{out_filename}>, skipping...\n")
                    continue

                else:
                    print(f"Found {len(output_items)} output(s) in <{out_filename}>\n")
                    preference_datas = []
                    for item in tqdm(output_items):
                        # get formatted prompt and completion
                        result = item['result']
                        if not result:
                            print(f"No result found for current item in <{out_filename}>, skipping...\n")
                            continue
                        formatted_prompt = result['formatted_prompt']
                        completions = result['completions']

                        # extract rejected and chosen samples
                        sorted_completions = sorted(completions, key=lambda x: x['reward'])
                        if sorted_completions[0]['reward'] < sorted_completions[-1]['reward']:
                            rejected_sample = sorted_completions[0]['formatted_sample']
                            chosen_sample = sorted_completions[-1]['formatted_sample']
                            preference_data = {
                                'prompt': formatted_prompt,
                                'chosen': process_sample(formatted_prompt, chosen_sample, domain),
                                'rejected': process_sample(formatted_prompt, rejected_sample, domain),
                            }
                            preference_datas.append(preference_data)

                    # write domain data
                    preference_filename = f"{output_source_dir}/{dpo_filename}"
                    dump_json(preference_filename, preference_datas)
                    all_preference_datas.extend(preference_datas)

            # write all data
            if args.do_aggregate:
                all_output_dir = f"generate/Common/All/{args.backend.replace('/', '--')}"
                if not os.path.exists(all_output_dir):
                    os.makedirs(all_output_dir)
                all_preference_filename = f"{all_output_dir}/{dpo_filename}"
                dump_json(all_preference_filename, all_preference_datas)

        elif args.mode == "sft":
            all_finetune_datas = []
            sft_filename = f"temp_{args.temperature}_tokens_{args.max_tokens}_sft_data.jsonl"

            for domain in domains:
                output_source_dir = f"generate/Common/{domain}/{args.backend.replace('/', '--')}"
                out_filename = f"{output_source_dir}/temp_{args.temperature}_tokens_{args.max_tokens}_completions.jsonl"
                output_items = read_json(out_filename)
                if len(output_items) == 0:
                    print(f"No output found in <{out_filename}>, skipping...\n")
                    continue

                else:
                    print(f"Found {len(output_items)} output(s) in <{out_filename}>\n")
                    finetune_datas = []
                    for item in tqdm(output_items):
                        # get formatted prompt and completion
                        result = item['result']
                        if not result:
                            print(f"No result found for current item in <{out_filename}>, skipping...\n")
                            continue
                        formatted_prompt = result['formatted_prompt']
                        completions = result['completions']

                        # extract accepted samples
                        for completion in completions:
                            formatted_sample = completion['formatted_sample']
                            reward = completion['reward']
                            if reward < args.completion_accept_threshold_sft:
                                continue
                            finetune_data = {
                                'prompt': formatted_prompt,
                                'completion': process_sample(formatted_prompt, formatted_sample, domain),
                            }
                            finetune_datas.append(finetune_data)

                    # write domain data
                    finetune_filename = f"{output_source_dir}/{sft_filename}"
                    dump_json(finetune_filename, finetune_datas)
                    all_finetune_datas.extend(finetune_datas)

            # write all data
            if args.do_aggregate:
                all_output_dir = f"generate/Common/All/{args.backend.replace('/', '--')}"
                if not os.path.exists(all_output_dir):
                    os.makedirs(all_output_dir)
                all_finetune_filename = f"{all_output_dir}/{sft_filename}"
                dump_json(all_finetune_filename, all_finetune_datas)

        else:
            raise NotImplementedError

    elif args.method == "MCTS":
        all_reward_datas = []

        for domain in domains:
            output_source_dir = f"generate/MCTS/{domain}/{args.backend.replace('/', '--')}"
            out_filename = f"{output_source_dir}/time_{args.time_limit}_iter_{args.iteration_limit}_sample_{args.num_sample}_decision_{args.num_decision}_exp_{args.exploration_constant}_eps_{args.eps}_temp_{args.temperature}_tokens_{args.max_tokens}_thought_{'yes' if args.use_thought else 'no'}.jsonl"
            output_items = read_json(out_filename)
            if len(output_items) == 0:
                print(f"No output found in <{out_filename}>, skipping...\n")
                continue

            else:
                print(f"Found {len(output_items)} output(s) in <{out_filename}>\n")
                reward_datas = []
                for item in tqdm(output_items):
                    # get prompt
                    prompt = item['prompt']
                    init_state = item['init_state']
                    completions = item['completions']

                    # terminal state rewards
                    for completion in completions:
                        new_reward_data = {'prompt': prompt, 'answer': completion['completion'],
                                           'reward': completion['reward']}
                        reward_datas.append(new_reward_data)

                    # intermediate state rewards
                    rewards = item['rewards']
                    for reward in rewards:
                        if reward['expanded']:
                            new_reward_data = {'prompt': prompt, 'answer': reward['state'], 'reward': reward['reward']}
                            reward_datas.append(new_reward_data)

                # write domain data
                rewards_filename = f"{output_source_dir}/time_{args.time_limit}_iter_{args.iteration_limit}_sample_{args.num_sample}_decision_{args.num_decision}_exp_{args.exploration_constant}_eps_{args.eps}_temp_{args.temperature}_tokens_{args.max_tokens}_thought_{'yes' if args.use_thought else 'no'}_rewards.jsonl"
                dump_json(rewards_filename, reward_datas)
                all_reward_datas.extend(reward_datas)

        # write all data
        if args.do_aggregate:
            all_output_dir = f"generate/MCTS/All/{args.backend.replace('/', '--')}"
            if not os.path.exists(all_output_dir):
                os.makedirs(all_output_dir)
            all_rewards_filename = f"{all_output_dir}/time_{args.time_limit}_iter_{args.iteration_limit}_sample_{args.num_sample}_decision_{args.num_decision}_exp_{args.exploration_constant}_eps_{args.eps}_temp_{args.temperature}_tokens_{args.max_tokens}_thought_{'yes' if args.use_thought else 'no'}_domains_{'_'.join(domains)}_rewards.jsonl"
            dump_json(all_rewards_filename, all_reward_datas)


if __name__ == "__main__":
    arguments = parse_args()
    process(arguments)
