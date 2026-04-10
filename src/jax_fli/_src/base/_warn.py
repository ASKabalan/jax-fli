import sys
import warnings

import jax
import jax.lax as lax

# ---------------------------------------------------------------------------
# ANSI colours (disabled when stderr is not a TTY)
# ---------------------------------------------------------------------------

_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

if not sys.stderr.isatty():
    _YELLOW = _RED = _BOLD = _RESET = ""


def warn_disabled(feature: str, workaround: str = "") -> None:
    """Issue a prominent, coloured warning that a feature is disabled.

    Inspired by the CADRE logging style (CMBSciPol/CADRE).
    Uses ``warnings.warn`` so the message is de-duplicated by Python's
    warning machinery and respects ``-W`` / ``PYTHONWARNINGS``.

    Parameters
    ----------
    feature:
        Short name of the disabled feature (e.g. ``"drift_on_lightcone"``).
    workaround:
        Optional one-liner telling the user what to do instead.
    """
    border = f"{_YELLOW}{_BOLD}" + "!" * 70 + _RESET
    header = f"{_RED}{_BOLD}  ⚠  FEATURE DISABLED: {feature}{_RESET}"
    body = f"{_YELLOW}  This feature is currently deactivated and has no effect.{_RESET}"
    lines = ["\n", border, header, body]
    if workaround:
        lines.append(f"{_YELLOW}  → {workaround}{_RESET}")
    lines.append(border)
    warnings.warn("\n".join(lines), UserWarning, stacklevel=3)


def _issue_warning(msg, *args):
    """The python function that runs on the host."""
    warnings.warn(msg.format(*args), stacklevel=4)


def warning_if(x, pred, msg, *args):
    """
    1. Checks 'pred' on the device (GPU/TPU).
    2. If True, triggers a callback to the host to print the warning.
    3. Returns 'x' unchanged (identity).

    Parameters
    ----------
    x : any
        Passthrough value (returned unchanged).
    pred : bool-like
        Traced predicate evaluated on device.
    msg : str
        A ``str.format()``-style template (e.g. ``"value is {:.2f}"``).
    *args
        Traced JAX values forwarded through ``jax.debug.callback``
        and materialized on the host before formatting.
    """

    def _trigger_warn(_):
        jax.debug.callback(_issue_warning, msg, *args)

    lax.cond(pred, _trigger_warn, lambda _: None, operand=None)
    return x
