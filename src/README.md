## Environment setup
```bash
conda create -n graphit python=3.10
conda activate graphit
pip install -e .
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.33.2
pip install accelerate==0.23.0
pip install trl
pip install deepspeed
pip install datasets
pip install tensorboard
```
## Flash Attention
For users who want to train very long sequence length, please also install FlashAttention:
```bash
pip install packaging
pip install ninja
ninja --version
echo $?
```
Make sure you see the output from echo is 0.
```bash
pip install flash-attn==2.1.1 --no-build-isolation
```
This installation is a bit slow. Plz wait.
## Finetuning
### Add your huggingface token to the HUGGING_FACE_TOKEN in longchat/train/fine_tune/train.py
To train a LLama2-13b-hf with 16k context length:
```bash
sh run_finetune.sh
```
This script assumes 8xA100 GPUs (40GB) and use the sharedGPT data. Note the sharedGPT data is the unfiltered version with very basic clean up. Please adapt to your use case.
## Finetuning with DeepSpeed
To train a LLama2-7b-hf with 16k context length with deepspeed stage:
```bash
sh run_deepspeed_finetune.sh
```
This script uses more memory but is faster comparing to training without deepspeed.
## Pretraining
```bash
sh run_pretrain.sh
```
## Pretraining with DeepSpeed
To train a LLama2-7b-hf with 16k context length with deepspeed stage:
```bash
sh run_deepspeed_pretrain.sh
```
This script uses more memory but is faster comparing to pretraining without deepspeed.

## Inference with TGI
```bash
pip install text-generation
```
```bash
sh tgi_local.sh
```
This script uses more memory but is faster comparing to pretraining without deepspeed.




