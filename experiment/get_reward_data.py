import os
import argparse
import numpy as np
from utils.json_operator import *


def parse_args():
    parser = argparse.ArgumentParser(description="Get reward distribution data")
    parser.add_argument("--domain", type=str, default="All", choices=["BigCodeBench", "DS1000", "APPS", "All"],
                        help="Specify the dataset")
    parser.add_argument("--backend", type=str, default="gpt-4o-mini", help="Specify the policy backend")
    parser.add_argument("--temperature", type=float, default=0.7, help="Specify the temperature for sampling")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Specify the maximum number of tokens available for generate")
    parser.add_argument("--do_not_format", action='store_true', default=False,
                        help="Specify whether not to format prompts")

    args = parser.parse_args()
    return args


def run(args):
    if args.domain == "All":
        domains = ["BigCodeBench", "DS1000", "APPS"]
    else:
        domains = [args.domain]

    all_rewards = []
    all_reward_stds = []
    for domain in domains:
        output_dir = f"generate/Common/{domain}/{args.backend.replace('/', '--')}"
        if not args.do_not_format:
            reward_filename = f"{output_dir}/temp_{args.temperature}_tokens_{args.max_tokens}_completions.jsonl"
        else:
            reward_filename = f"{output_dir}/temp_{args.temperature}_tokens_{args.max_tokens}_completions_no_format.jsonl"

        datas = read_json(reward_filename)

        rewards = []
        reward_stds = []
        for data in datas:
            result = data["result"]
            if not result:
                continue
            completions = result["completions"]
            cur_rewards = [x["reward"] for x in completions]
            rewards.extend(cur_rewards)
            reward_stds.append(np.std(cur_rewards))

        # visualize domain data
        visualize_dir = f"logs/{domain}/{args.backend.replace('/', '--')}"
        if not os.path.exists(visualize_dir):
            os.makedirs(visualize_dir)
        if not args.do_not_format:
            visualize_reward_filename = f"{visualize_dir}/temp_{args.temperature}_tokens_{args.max_tokens}_rewards.png"
            visualize_reward_title = f"Reward Distribution for {domain} (Naive GRPO)"
            visualize_std_filename = f"{visualize_dir}/temp_{args.temperature}_tokens_{args.max_tokens}_stds.png"
            visualize_std_title = f"Reward Standard Deviation Distribution for {domain} (Naive GRPO)"
        else:
            visualize_reward_filename = f"{visualize_dir}/temp_{args.temperature}_tokens_{args.max_tokens}_no_format_rewards.png"
            visualize_reward_title = f"Reward Distribution for {domain} (ReST-GRPO)"
            visualize_std_filename = f"{visualize_dir}/temp_{args.temperature}_tokens_{args.max_tokens}_no_format_stds.png"
            visualize_std_title = f"Reward Standard Deviation Distribution for {domain} (ReST-GRPO)"

        print(f"{domain} - Average Reward Std: {np.mean(reward_stds):.4f}")
        visualize_reward_distribution(rewards, visualize_reward_title, visualize_reward_filename)
        visualize_std_distribution(reward_stds, visualize_std_title, visualize_std_filename)

        # add to all data
        all_rewards.extend(rewards)
        all_reward_stds.extend(reward_stds)

    # visualize all data
    visualize_dir = f"logs/All/{args.backend.replace('/', '--')}"
    if not os.path.exists(visualize_dir):
        os.makedirs(visualize_dir)
    if not args.do_not_format:
        visualize_reward_filename = f"{visualize_dir}/temp_{args.temperature}_tokens_{args.max_tokens}_rewards.png"
        visualize_reward_title = "Reward Distribution (Naive GRPO)"
        visualize_std_filename = f"{visualize_dir}/temp_{args.temperature}_tokens_{args.max_tokens}_stds.png"
        visualize_std_title = "Reward Standard Deviation Distribution (Naive GRPO)"
    else:
        visualize_reward_filename = f"{visualize_dir}/temp_{args.temperature}_tokens_{args.max_tokens}_no_format_rewards.png"
        visualize_reward_title = "Reward Distribution (ReST-GRPO)"
        visualize_std_filename = f"{visualize_dir}/temp_{args.temperature}_tokens_{args.max_tokens}_no_format_stds.png"
        visualize_std_title = "Reward Standard Deviation Distribution (ReST-GRPO)"

    print(f"All Domains - Average Reward Std: {np.mean(all_reward_stds):.4f}")
    visualize_reward_distribution(all_rewards, visualize_reward_title, visualize_reward_filename)
    visualize_std_distribution(all_reward_stds, visualize_std_title, visualize_std_filename)


def visualize_reward_distribution(reward_list, title, output_path="reward_distribution.png"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not reward_list:
        print(f"No data to plot for {title}")
        return

    mean_reward = np.mean(reward_list)
    median_reward = np.median(reward_list)
    std_reward = np.std(reward_list)

    plt.figure(figsize=(12, 7), dpi=300)

    sns.histplot(reward_list, bins=50, kde=True, color='skyblue', edgecolor='black', alpha=0.7, stat='density')

    plt.axvline(mean_reward, color='darkorange', linestyle='--', linewidth=6, label=f'Mean: {mean_reward:.4f}')
    plt.axvline(median_reward, color='green', linestyle='--', linewidth=6, label=f'Median: {median_reward:.4f}')
    plt.axvline(mean_reward + std_reward, color='gray', linestyle=':', linewidth=6, label=f'SD: ±{std_reward:.4f}')
    plt.axvline(mean_reward - std_reward, color='gray', linestyle=':', linewidth=6)

    plt.title(title, fontsize=24, fontweight='bold')
    plt.xlabel("Reward Value", fontsize=24)
    plt.ylabel("Density", fontsize=24)

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.legend(loc='upper right', fontsize=24)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def visualize_std_distribution(std_list, title, output_path="reward_std_distribution.png"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not std_list:
        print(f"No data to plot for {title}")
        return

    mean_std = np.mean(std_list)
    median_std = np.median(std_list)

    plt.figure(figsize=(12, 7), dpi=300)

    sns.histplot(std_list, bins=50, kde=True, color='salmon', edgecolor='black', alpha=0.7, stat='density')

    plt.axvline(mean_std, color='darkred', linestyle='--', linewidth=6, label=f'Mean: {mean_std:.4f}')
    plt.axvline(median_std, color='steelblue', linestyle='--', linewidth=6, label=f'Median: {median_std:.4f}')

    plt.title(title, fontsize=24, fontweight='bold')
    plt.xlabel("Standard Deviation of Rewards", fontsize=24)
    plt.ylabel("Density", fontsize=24)

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.legend(loc='upper right', fontsize=24)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    run(args)
