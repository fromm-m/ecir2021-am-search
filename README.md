# Argument Clustering

Accompanying repository of our [ECIR2021](https://arxiv.org/abs/2011.02177) paper "Diversity Aware Relevance Learning for Argument Search".

[![Python 3.8](https://img.shields.io/badge/Python-3.8-2d618c?logo=python)](https://docs.python.org/3.8/)
[![PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-ee4c2c?logo=pytorch)](https://pytorch.org/docs/stable/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

We used the dataset of [ECIR2020](https://link.springer.com/chapter/10.1007/978-3-030-45439-5_29) paper "A Framework for Argument Retrieval".

Install requirements (in this virtual environment):
```bash
pip install -U pip
pip install -U -r requirements.txt
```

Execution
```bash
PYTHONPATH=src:$PYTHONPATH python3 executables/evaluation/evaluate_baselines.py --force > output/output_energy.txt
```
