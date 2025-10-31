#!/bin/bash
set -x

# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=cache/huggingface

port=$(shuf -i25000-30000 -n1)
# --nproc_per_node=1
# CUDA_VISIBLE_DEVICES=1,4,7 python -m torch.distributed.run --nproc_per_node=3 \

deepspeed --master_port $port  --include=localhost:4,5,6  \
        longchat/train/fine_tune/finetune.py \
        --model_name_or_path checkpoint/llama2-7b-hf/checkpoint-1 \
        --data_path data/mix_train.json  \
        --bf16 \
        --output_dir checkpoint/llama2_7b_lora_mix \
        --num_train_epochs 5    \
        --per_device_train_batch_size 3 \
        --per_device_eval_batch_size 1  \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy no \
        --save_strategy epoch \
        --save_steps 1 \
        --save_total_limit 5 \
        --learning_rate 1e-4 \
        --weight_decay 0.  \
        --warmup_ratio 0.03  \
        --lr_scheduler_type "cosine" \
        --logging_steps 1  \
        --model_max_length 1024  \
        --lazy_preprocess True \
        --report_to tensorboard \
        --tf32 True \
        --deepspeed ./deepspeed_config/stage2.json  
        # --fsdp "full_shard auto_wrap" \
        # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        # --gradient_checkpointing True  
