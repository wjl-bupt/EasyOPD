"""OPLD: On-Policy Listwise Distillation.

The student samples K rollouts per prompt. Teacher and student each turn their
K per-sequence logprobs into a group-wise categorical distribution via softmax,
and the teacher distribution is distilled into the student one with a KL. That
group KL's exact gradient is a per-sequence policy gradient, so OPLD is
implemented as an *advantage estimator* rather than a distillation loss.

Integration mode: advantage estimator (driver-side).
    - ``@register_adv_est("listwise")`` into verl's ADV_ESTIMATOR_REGISTRY
    - a ``compute_advantage`` wrapper that reduces teacher/student per-token
      logprobs to sequence level before calling the estimator

Modified verl files: none. The wrapper is installed at runtime from
``register()`` so ``verl/`` stays untouched.

Usage (yaml)::

    algorithm:
      adv_estimator: listwise
      easyopd:
        listwise:
          beta: 1.0
          length_norm: true
    actor_rollout_ref:
      rollout:
        n: 8          # MUST be > 1: listwise needs a list

Public surface:
    * OPLD / METHOD                            (this module)
    * compute_listwise_advantage               (core.py)
    * read_listwise_config                     (core.py)
    * compute_listwise_advantage_estimator     (advantage_estimator.py)
"""

from dataclasses import dataclass

from easyopd.registry import register_method


@register_method("listwise", loss_mode_aliases=("opld",))
@dataclass(frozen=True)
class OPLD:
    """Static metadata describing the EasyOPD ``listwise`` (OPLD) method."""

    name: str = "listwise"
    description: str = (
        "On-policy listwise distillation: distills the teacher's group-softmax "
        "over K rollouts of a prompt into the student's, via the equivalent "
        "per-sequence advantage A_i = q_T(i) - q_S(i)."
    )
    paper_url: str = "coming soon"
    verl_modified_files: tuple = ()
    capabilities: tuple = ("advantage_estimator",)
    integration_mode: str = "advantage-estimator"


METHOD = OPLD()


def register() -> None:
    """Register OPLD's advantage estimator and install the driver-side patch.

    Importing ``advantage_estimator`` is what actually runs
    ``@register_adv_est("listwise")``; the patch then feeds that estimator the
    tensors verl's generic branch does not forward.
    """
    # Imported lazily so that importing this package does not pull torch/verl
    # until registration is actually attempted.
    from easyopd.methods.opld.advantage_estimator import install_compute_advantage_patch

    install_compute_advantage_patch()


# NOTE: nothing in the framework calls ``register()`` -- ``auto_discover`` only
# imports this package's ``__init__``. A purely lazy registration would mean the
# estimator is never in ADV_ESTIMATOR_REGISTRY by the time
# ``get_adv_estimator_fn("listwise")`` runs, so we self-register on import.
# Guarded because discovery also happens in environments without torch/verl
# installed, where we still want the method metadata to register.
try:
    register()
except Exception as _err:  # noqa: BLE001
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "[EasyOPD:opld] advantage estimator not registered (%s: %s). "
        "The method metadata is registered, but training with "
        "algorithm.adv_estimator=listwise will fail until this import succeeds.",
        type(_err).__name__,
        _err,
    )


def __getattr__(name: str):
    """Lazily expose the public surface without importing verl on package import."""
    if name in {"compute_listwise_advantage", "read_listwise_config"}:
        from easyopd.methods.opld import core

        return getattr(core, name)

    if name in {
        "compute_listwise_advantage_estimator",
        "install_compute_advantage_patch",
        "OPLDMissingTeacherLogprobs",
        "OPLDMissingUID",
        "OPLDConflictingKLTerm",
    }:
        from easyopd.methods.opld import advantage_estimator

        return getattr(advantage_estimator, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "METHOD",
    "OPLD",
    "register",
    "compute_listwise_advantage",
    "read_listwise_config",
    "compute_listwise_advantage_estimator",
    "install_compute_advantage_patch",
    "OPLDMissingTeacherLogprobs",
    "OPLDMissingUID",
    "OPLDConflictingKLTerm",
]
