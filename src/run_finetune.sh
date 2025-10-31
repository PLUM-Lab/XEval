#!/bin/bash
set -x

# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=cache/huggingface

port=$(shuf -i25000-30000 -n1)
# --nproc_per_node=1
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 \
        longchat/train/fine_tune/finetune.py \
        --model_name_or_path checkpoint/llama2_7b_stage1_full \
        --data_path data/stage2_train.json  \
        --bf16 \
        --output_dir checkpoint/llama2_7b_stage2_full \
        --num_train_epochs 1    \
        --per_device_train_batch_size 3 \
        --per_device_eval_batch_size 1  \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 500 \
        --save_total_limit 2 \
        --learning_rate 1e-5 \
        --weight_decay 0.  \
        --warmup_ratio 0.03  \
        --lr_scheduler_type "cosine" \
        --logging_steps 1  \
        --model_max_length 1024  \
        --lazy_preprocess True \
        --report_to tensorboard \
        --tf32 True \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --gradient_checkpointing True  
