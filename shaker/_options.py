"""SHAKER-wide defaults for per-call arguments left as None."""

_OPTIONS = {"show": True}


def set_options(show=None):
    """
    Set SHAKER-wide defaults, used wherever the matching argument is None.

    An argument passed explicitly to a function always wins over these.

    Parameters
    ----------
    show : bool, optional
        Whether plotting functions display their figures in the notebook.
        Figures are still saved to disk when an output file is set. Default
        at import is True; e.g. ``set_options(show=False)`` keeps notebook
        output small when an agent is driving SHAKER.
    """
    if show is not None:
        _OPTIONS["show"] = bool(show)


def _resolve(value, key):
    """`value` if given, else the SHAKER-wide default for `key`."""
    return _OPTIONS[key] if value is None else value
