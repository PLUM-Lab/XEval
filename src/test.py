from transformers import AutoTokenizer, AutoModelForCausalLM
HUGGING_FACE_TOKEN = ""




# model_path = "llama2_7b_chat_stage1/checkpoint-4191"
model_path = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=HUGGING_FACE_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token=HUGGING_FACE_TOKEN)
model.half().cuda()

prompt = """\
### Human: Write a Python script for text classification using Transformers and PyTorch
### Assistant:\
"""

inputs = tokenizer(prompt, return_tensors='pt').to('cuda')


tokens = model.generate(
 **inputs,
 max_new_tokens=256,
 do_sample=True,
 temperature=1.0,
 top_p=1.0,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
