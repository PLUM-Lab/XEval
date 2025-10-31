from transformers import AutoModelForCausalLM
import torch
import pdb

model_path = "checkpoint/llama2_7b_stage1_epo1/checkpoint-6000"

device = torch.device("cuda:3")

# Load the model from a checkpoint directory
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to(device)

# Function to check if a tensor contains any NaN, infinite or unexpected values
def contains_anonymous_values(tensor):
    # Check for NaN values
    if torch.isnan(tensor).any():
        # print("------parameter contain nan-----")
        return True
    # Check for infinite values
    if torch.isinf(tensor).any():
        # print("------parameter contain inf-----")
        return True
    # Add more checks here if needed for other "anonymous" values
    return False

# Iterate through model parameters and check for anonymous values
for name, param in model.named_parameters():
    if contains_anonymous_values(param):
        print(name)
        # print(f"Parameter '{name}' contains anonymous values.")
        # pdb.set_trace()
