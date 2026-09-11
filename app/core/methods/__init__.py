"""The seven DA methods of SPEC §10, fork 5. Do not add an eighth without asking (§21)."""
from __future__ import annotations

from . import deseq
from .aldex2 import run_aldex2
from .ancombc import run_ancombc
from .deseq import run_pydeseq2
from .elementary import run_linear, run_logistic, run_ttest, run_wilcoxon

METHOD_REGISTRY = {
    "wilcoxon": run_wilcoxon,
    "ttest": run_ttest,
    "logistic": run_logistic,
    "linear": run_linear,
    "ancombc": run_ancombc,
    "aldex2": run_aldex2,
    "pydeseq2": run_pydeseq2,
}

#: Methods that model covariates directly rather than through residualisation.
COVARIATE_NATIVE = {"linear", "logistic", "ancombc", "pydeseq2"}


def available_methods() -> dict:
    """Method -> reason it is unavailable ('' when it can run)."""
    status = dict.fromkeys(METHOD_REGISTRY, "")
    if not deseq.available():
        status["pydeseq2"] = "pydeseq2 is not installed in this environment"
    return status


def run_method(name: str, matrix, covariate_frame=None, *, aldex_instances: int = None):
    """Dispatch to one method. Options a method cannot use are dropped here rather
    than being threaded through every signature."""
    try:
        runner = METHOD_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown DA method: {name}") from exc
    if name == "aldex2" and aldex_instances:
        return runner(matrix, covariate_frame, n_instances=int(aldex_instances))
    return runner(matrix, covariate_frame)
