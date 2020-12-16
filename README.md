# Argument Clustering

Accompanying repository of our [ECIR2021](https://arxiv.org/abs/2011.02177) paper "Diversity Aware Relevance Learning for Argument Search".

[![Python 3.8](https://img.shields.io/badge/Python-3.8-2d618c?logo=python)](https://docs.python.org/3.8/)
[![PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-ee4c2c?logo=pytorch)](https://pytorch.org/docs/stable/index.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

We used the dataset of the [ECIR2020](https://link.springer.com/chapter/10.1007/978-3-030-45439-5_29) paper "A Framework for Argument Retrieval", which is available by request from the authors.

## Install requirements:
```bash
pip install -U pip
pip install -U -r requirements.txt
```

## Preprocessing:
You have to request the dataset from the authors and adjust the paths in [settings.py](src/arclus/settings.py)
### Step 1: Extract all claims, premises and the matching
```bash
PYTHONPATH=src:$PYTHONPATH python3 executables/preprocessing/read_json.py --input_dir=... --output_dir=output/
```

### Step 2: Precompute bert-features for the claims, premises and claim-premise pairs (choices=['pair', 'claims', 'premises'])
```bash
PYTHONPATH=src:$PYTHONPATH python3 executables/preprocessing/generate_features.py --mode=...
```

### Step 3: Generate negative claim-premise pairs either [randomly](executables/preprocessing/generate_negative_samples.py) or based on [similarity](executables/preprocessing/generate_negative_samples_nn.py)
```bash
# Either
PYTHONPATH=src:$PYTHONPATH python3 executables/preprocessing/generate_negative_samples_nn.py
# or
PYTHONPATH=src:$PYTHONPATH python3 executables/preprocessing/generate_negative_samples.py
```
### Step 4: Generate train-/test-/validation split
```bash
PYTHONPATH=src:$PYTHONPATH python3 executables/preprocessing/generate_sets.py
```

## Execution:
```bash
PYTHONPATH=src:$PYTHONPATH python3 executables/evaluation/evaluate_baselines.py --force > output/output_energy.txt
```
