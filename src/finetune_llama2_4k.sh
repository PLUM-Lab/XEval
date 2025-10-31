export DATASET=$1
export OUTDIR=$2
export TRANSFORMERS_CACHE=/projects/nlp_lab/zhiyang/.cache # TODO: change to your path

python -m torch.distributed.run --nproc_per_node=2 \
         longchat/train/fine_tune/finetune.py \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --data_path ./data/$DATASET  \
        --do_train \
        --bf16 \
        --output_dir checkpoints/$OUTDIR \
        --num_train_epochs 3    \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 1  \
        --gradient_accumulation_steps 1 \
        --save_strategy epoch \
        --evaluation_strategy no \
        --save_total_limit 3 \
        --learning_rate 2e-5 \
        --weight_decay 0.  \
        --warmup_ratio 0.03  \
        --lr_scheduler_type "cosine" \
        --logging_steps 1  \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True  \
        --model_max_length 4096  \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --report_to tensorboard
