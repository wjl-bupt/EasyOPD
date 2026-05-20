# SOD: Step-wise On-policy Distillation

## Method Overview

SOD addresses the instability of standard OPD in tool-integrated reasoning (TIR) scenarios.
It adaptively re-weights the distillation strength at each reasoning step based on the divergence trajectory.

**Paper:** https://arxiv.org/abs/2605.07725

## Key Equations

- Eq. 6 (Step Divergence): d_k = mean(|log pi_theta - log pi_teacher|) over step k
- Eq. 7 (Adaptive Weight): w_k = min(prod (d_u+eps)/(d_{u+1}+eps), 1+delta)
- Eq. 10 (Training Objective): L = L_GRPO + opd_coef * w_k * (log pi_teacher - log pi_theta)

## Modifications to verl

- verl/trainer/config/algorithm.py: Added TokenKLRegConfig dataclass
- verl/trainer/ppo/ray_trainer.py: Added _apply_token_kl_regularizer method

## Hyperparameters

- token_kl_reg.enable: Enable the token KL regularizer module (default: False)
- token_kl_reg.stepwise_enable: Enable step-wise mode (default: False)
- token_kl_reg.stepwise_epsilon: Numerical stability constant (default: 1e-6)
- token_kl_reg.stepwise_delta: Upper bound offset (default: 0.5)
- token_kl_reg.stepwise_opd_coef: Global OPD coefficient (default: 1.0)

## Reproduction

bash examples/sod/run_sod.sh
