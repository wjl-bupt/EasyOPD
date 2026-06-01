# EasyOPD: A Unified Framework for On-Policy Distillation

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-Paper-red.svg" alt="Paper"></a>
  <a href="https://github.com/lds-ustc/EasyOPD"><img src="https://img.shields.io/badge/GitHub-Code-blue.svg" alt="Code"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick-Start-green.svg" alt="Quick Start"></a>
</p>

**EasyOPD** is a unified framework for On-Policy Distillation (OPD) built on top of [verl](https://github.com/verl-project/verl). It provides a single, consistent interface to run diverse OPD methods — from standard logit-based KD to cross-tokenizer distillation, agentic step-wise OPD, and self-distillation — by simply switching a YAML config file.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Layer                                │
│   YAML Config  ──→  EasyOPD.from_hparams("method_name")        │
│                      scripts/run_easyopd.sh --method gkd        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                      EasyOPD Layer                               │
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────┐ │
│  │ Registry │  │ Hook Dispatch│  │Config Loader│  │Diagnostics│ │
│  │          │  │              │  │             │  │           │ │
│  │@register │  │ LossHook     │  │ YAML merge  │  │ Metrics   │ │
│  │from_hpara│  │ RolloutHook  │  │ Validation  │  │ Anomaly   │ │
│  │auto_disc │  │ RewardHook   │  │ Defaults    │  │ Reporting │ │
│  │          │  │ AlignHook    │  │             │  │           │ │
│  │          │  │ TeacherHook  │  │             │  │           │ │
│  └──────────┘  └──────────────┘  └────────────┘  └──────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    methods/                                  │ │
│  │  simple │ simct │ gkd │ sod │ g_opd │ opcd │ vision_opd │ sdpo │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                        verl Layer                                │
│   Ray-based distributed training │ FSDP/Megatron │ vLLM/SGLang  │
│   PPO Trainer │ Actor/Critic Workers │ Reward Model              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Supported Methods

| Method | Regime | Teacher Type | Key Feature | Paper |
|--------|--------|-------------|-------------|-------|
| **simple** | Cross-tokenizer OPD | External (different tokenizer) | Overlap vocabulary KL with character-level alignment | — |
| **simct** | Cross-tokenizer OPD | External (different tokenizer) | Span-based virtual vocabulary logits | [SimCT](https://arxiv.org/abs/xxxx.xxxxx) |
| **gkd** | Standard OPD | External (same tokenizer) | Generalized JSD with on-policy sampling | [GKD (ICLR'24)](https://arxiv.org/abs/2306.13649) |
| **sod** | Agentic OPD | External | Step-wise adaptive re-weighting for TIR agents | [SOD](https://arxiv.org/abs/2605.07725) |
| **g_opd** | Generalized OPD | External + Context | Reward extrapolation + flexible reference model | [G-OPD](https://arxiv.org/abs/2602.12125) |
| **opcd** | Context Distillation | Self (context-conditioned) | Experience injection + KL minimization | [OPCD](https://arxiv.org/abs/2602.12275) |
| **vision_opd** | Multimodal Self-OPD | Self (EMA + fine-grained visual) | EMA teacher with bbox-cropped images | [Vision-OPD](https://arxiv.org/abs/2605.18740) |
| **sdpo** | Self-Distillation | Self (EMA/reprompt) | Self-distillation from high-reward trajectories | [SDPO](https://arxiv.org/abs/2601.20802) |

**Planned:**
- ROPD (Rubric-based Black-box OPD)
- DSKD (Dual-Space Knowledge Distillation)
- ALM (Adaptive Logit Matching)
- ULD (Universal Logit Distillation)

---

## Quick Start

### Installation

```bash
git clone https://github.com/lds-ustc/EasyOPD.git
cd EasyOPD
pip install -e .
```

### List Available Methods

```bash
python scripts/run_easyopd.py --list-methods
```

### Run a Method

```bash
# Using the unified launch script
bash scripts/run_easyopd.sh --method gkd --config easyopd/config/gkd.yaml

# Or directly with Python
python scripts/run_easyopd.py --method simple --config easyopd/config/simple.yaml

# Dry run (validate config without training)
python scripts/run_easyopd.py --method sod --dry-run
```

### Python API

```python
from easyopd import EasyOPD

# Discover available methods
print(EasyOPD.list_methods())
# ['g_opd', 'gkd', 'opcd', 'sdpo', 'simct', 'simple', 'sod', 'vision_opd']

# Load a method with config
method = EasyOPD.from_hparams("gkd", config_path="easyopd/config/gkd.yaml")
print(method.description)
print(method.paper_url)
```

---

## Configuration

Each method has a default YAML config in `easyopd/config/`. To customize:

```yaml
# my_experiment.yaml
defaults:
  - _self_

# Override method-specific parameters
distillation:
  enabled: true
  distillation_loss:
    loss_mode: gkd
    gkd_beta: 0.5  # JSD interpolation (0=forward KL, 1=reverse KL)

# Model paths
student_model_path: Qwen/Qwen3-8B
teacher_model_path: Qwen/Qwen3-72B

# Training
train_batch_size: 64
actor_lr: 1.0e-6
```

Switch methods by changing `loss_mode` and the corresponding parameters:

```bash
# GKD
python scripts/run_easyopd.py --method gkd --config my_gkd_config.yaml

# Switch to SOD (just change the config)
python scripts/run_easyopd.py --method sod --config my_sod_config.yaml
```

---

## Project Structure

```
EasyOPD/
├── easyopd/                    # EasyOPD unified interface layer
│   ├── __init__.py             # EasyOPD class with from_hparams()
│   ├── registry.py             # Method registration (@register_method)
│   ├── hooks.py                # Hook Protocol interfaces
│   ├── hook_dispatch.py        # HookDispatcher (verl ↔ method routing)
│   ├── config_loader.py        # Unified config loading & validation
│   ├── diagnostics.py          # Metrics collection & anomaly detection
│   ├── config/                 # Default YAML configs per method
│   │   ├── simple.yaml
│   │   ├── gkd.yaml
│   │   └── ...
│   └── methods/                # Method implementations
│       ├── simple/             # Cross-tokenizer KD
│       ├── simct/              # Span-based cross-tokenizer KD
│       ├── gkd/                # Generalized Knowledge Distillation
│       ├── sod/                # Step-wise On-policy Distillation
│       ├── g_opd/              # Generalized OPD with reward extrapolation
│       ├── opcd/               # On-Policy Context Distillation
│       ├── vision_opd/         # Vision On-Policy Self-Distillation
│       └── sdpo/               # Self-Distilled Policy Optimization
├── verl/                       # verl framework (minimal modifications)
├── scripts/                    # Launch scripts
│   ├── run_easyopd.py          # Unified Python entry point
│   └── run_easyopd.sh          # Unified shell launcher
├── examples/                   # Method-specific training examples
├── tests/                      # Test suite
├── README.md                   # This file
├── README_VERL.md              # Original verl documentation
└── CONTRIBUTING.md             # Developer contribution guide
```

---

## Adding a New Method

1. Create a new directory under `easyopd/methods/your_method/`
2. Implement your core algorithm in `core.py`
3. Create hook adapters in `hooks.py` (implement the hooks you need)
4. Register with `@register_method("your_method")` in `__init__.py`
5. Add a default config in `easyopd/config/your_method.yaml`

```python
# easyopd/methods/your_method/__init__.py
from easyopd.registry import register_method

@register_method("your_method")
class YourMethod:
    name = "your_method"
    description = "Your OPD method description"
    paper_url = "https://arxiv.org/abs/..."
    verl_modified_files = []  # No verl modifications needed with hooks!
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guidelines.

---

## Citation

```bibtex
@article{easyopd2026,
  title={EasyOPD: A Unified Framework for On-Policy Distillation},
  author={...},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```

---

## Acknowledgments

EasyOPD is built on top of [verl](https://github.com/verl-project/verl) by the verl-project team. We thank the authors of all integrated OPD methods for their contributions to the field.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
