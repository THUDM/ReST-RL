deepspeed train.py \
  Qwen/Qwen2.5-Coder-7B-Instruct \
  ckpts/Qwen2.5-Coder-7B-Instruct/1 \
  your-train-file.jsonl \
  transformers_scalar \
  --max_length 2048 \
  --deepspeed_config config/zero2_config.json \
  --lr 1e-7 \
  --epochs 2 \
  --save_steps 0.5 \
  --batch_size_per_device 1