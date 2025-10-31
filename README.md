# X-Eval (NAACL 2024)

## Abstract

Natural Language Generation (NLG) typically involves evaluating the generated text in various aspects (e.g., consistency and naturalness) to obtain a comprehensive assessment. However, multi-aspect evaluation remains challenging as it may require the evaluator to generalize to any given evaluation aspect even if it's absent during training. In this paper, we introduce X-Eval, a two-stage instruction tuning framework to evaluate the text in both seen and unseen aspects customized by end users. X-Eval consists of two learning stages: the vanilla instruction tuning stage that improves the model's ability to follow evaluation instructions, and an enhanced instruction tuning stage that exploits the connections between fine-grained evaluation aspects to better assess text quality. To support the training of X-Eval, we collect AspectInstruct, the first instruction tuning dataset tailored for multi-aspect NLG evaluation spanning 27 diverse evaluation aspects with 65 tasks. To enhance task diversity, we devise an augmentation strategy that converts human rating annotations into diverse forms of NLG evaluation tasks, including scoring, comparison, ranking, and Boolean question answering. Extensive experiments across three essential categories of NLG tasks: dialogue generation, summarization, and data-to-text coupled with 21 aspects in meta-evaluation, demonstrate that our X-Eval enables even a lightweight language model to achieve a comparable if not higher correlation with human judgments compared to the state-of-the-art NLG evaluators, such as GPT-4.



## Repository Structure

```
XEval/
├── data/
│   ├── meta_eval/          # Meta-evaluation datasets
│   │   ├── data2text/      # Data-to-text evaluation (sfres, sfhot)
│   │   ├── dialogue/       # Dialogue evaluation (topical_chat)
│   │   ├── fact/           # Fact verification (qags_cnndm, qags_xsum)
│   │   └── summarization/  # Summarization evaluation (summeval)
│   └── tasks/
│       └── multi_hint/     # Multi-hint inference tasks
├── src/
│   ├── longchat/           # Core LongChat implementation
│   │   ├── conversation.py # Conversation templates and utilities
│   │   └── train/          # Training utilities and fine-tuning scripts
│   └── *.py               # Utility scripts and patches
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/XEval.git
cd XEval
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Overview

### Meta-Evaluation Datasets

1. **Data-to-Text Generation**
   - `sfres.json`: SFRes dataset for restaurant domain
   - `sfhot.json`: SFHot dataset for hotel domain
   - Metrics: informativeness, naturalness

2. **Dialogue Evaluation**
   - `topical_chat.json`: TopicalChat dataset
   - Metrics: coherence, understandability, naturalness, engagingness, groundedness

3. **Summarization**
   - `summeval.json`: SummEval dataset
   - Metrics: coherence, consistency, fluency, relevance

4. **Fact Verification**
   - `qags_cnndm.json`: QAGS evaluation on CNN/DM
   - `qags_xsum.json`: QAGS evaluation on XSum



## Usage

### Basic Evaluation

To evaluate a model on the provided datasets:

```python
from src.longchat.conversation import get_default_conv_template
import json

# Load evaluation data
with open('data/meta_eval/dialogue/topical_chat.json', 'r') as f:
    eval_data = json.load(f)

# Get conversation template
conv_template = get_default_conv_template("vicuna")

# Process your model's outputs using the conversation template
# (Add your model evaluation logic here)
```

### Fine-tuning

To fine-tune a model using the provided training script:

```bash
python src/longchat/train/fine_tune/train.py \
    --model_name_or_path facebook/opt-125m \
    --data_path path/to/your/training_data.json \
    --output_dir ./checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5 \
    --model_max_length 2048 \
    --lora_enable True \
    --lora_r 8 \
    --lora_alpha 16
```



## Configuration

The framework supports various model architectures and conversation templates:

- **Vicuna v1.1**: Default template for instruction-following
- **Koala**: Alternative conversation format
- **Dolly v2**: Instruction-response format
- **OASST**: OpenAssistant format
- **StableLM**: StabilityAI model format


## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{liu-etal-2024-x,
    title = "{X}-Eval: Generalizable Multi-aspect Text Evaluation via Augmented Instruction Tuning with Auxiliary Evaluation Aspects",
    author = "Liu, Minqian  and
      Shen, Ying  and
      Xu, Zhiyang  and
      Cao, Yixin  and
      Cho, Eunah  and
      Kumar, Vaibhav  and
      Ghanadan, Reza  and
      Huang, Lifu",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.473/",
    doi = "10.18653/v1/2024.naacl-long.473",
    pages = "8560--8579"
}

```

## Acknowledgments

This project builds upon:
- [LongChat](https://github.com/DachengLi1/LongChat) for long context capabilities
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) for training infrastructure
- Various evaluation datasets from the NLP community