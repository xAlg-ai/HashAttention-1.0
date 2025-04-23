# HashAttention : Semantic Sparsity for Faster Inference

[![arXiv.v1](https://img.shields.io/badge/arxiv.v1-2412.14468-B31B1B.svg)](https://arxiv.org/abs/2412.14468)

TLDR; HashAttention is a lightweight sparse attention.

## 📦 Installation


```
git clone https://github.com/xAlg-ai/hashattention1.0
pip install -e .
```

## 🔧 Key Features

- **🧠 Uses 32 bits per token auxiliary memory (lowest among competitors)
- **🧠 Upto 32x sparsity in LLAMA and MISTRAL 8B models.
- **🧠 Upto 4.3x improvement in GPT-fast attention latency
- **🧠 Upto 2.54x improvement in Flash-decode attention lateny
- **🧠 Upto 3.12x improvement in GPT-fast throughput



## 🚀 Quick Start

To play around with hashattention:

```python
from hashattention.hashattention_llama import convert
patch="./artifacts/llama3.1-8b-patch.64K.v1.pt"
model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
model  = AutoModelForCausalLM.from_pretrained(model_name_or_path)
config = AutoConfig.from_pretrained(model_name_or_path)
config.lth_init_dim = 128
config.lth_final_dim = 32
config.lth_thold = 0
config.init_budget = 128
config.heavy_budget = 0.25
config.recent_budget = 128
config.usa_eval_mode = "simple"
config.lth_num_layers = 3
usa_modules = load_usa(config, patch)
model = convert_usa(model, config, usa_modules, collect_stats = False, train_usa=False)
```


## 🧠 Contributions

This work is brought to you by HashAttention Team at sky-lab

Correspondance [Aditya Desai](https://apd10.github.io/)

## 📚 Citation

If you found our work useful or interesting, please consider citing our paper:

```bibtex
@article{desai2024hashattention,
  title={HashAttention: Semantic Sparsity for Faster Inference},
  author={Desai, Aditya and Yang, Shuo and Cuadron, Alejandro and Klimovic, Ana and Zaharia, Matei and Gonzalez, Joseph E and Stoica, Ion},
  journal={arXiv preprint arXiv:2412.14468},
  year={2024}
}
```

