accelerate launch train_grpo.py \
"Qwen/Qwen2.5-Coder-7B-Instruct" \
"ckpts/grpo/Qwen2.5-Coder-7B-Instruct/1" \
"your-train-file.jsonl" \
--max_prompt_length 1024 \
--max_completion_length 1024 \
--num_generations 8 \
--log_completions \
--deepspeed_config "config/zero2_config.json" \
--lr 1e-7 \
--epochs 1 \
--save_steps 0.5 \
--batch_size_per_device 2 \
--gradient_accumulation_steps 1 \
--symbol_reward 1e-3 \
--trailing_penalty 1e-6 \
--report_to "wandb"