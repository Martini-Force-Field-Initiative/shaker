"""
SHAKER-wide defaults for display and output.

Functions that plot or print take `show`, `verbose` (and use progress bars)
with a default of None, meaning "use the SHAKER-wide setting". `set_options`
changes those settings for the session; an explicit argument always wins.
"""

from tqdm.autonotebook import tqdm

_OPTIONS = {"show": True, "verbose": True, "progress": True}


def set_options(show=None, verbose=None, progress=None):
    """
    Set SHAKER-wide defaults, used wherever the matching argument is None.

    An argument passed explicitly to a function always wins over these.
    Arguments left as None here keep their current setting. They last until
    the kernel restarts.

    Parameters
    ----------
    show : bool, optional
        Whether plotting functions display their figures in the notebook.
        Figures are still saved to disk when an output file is set. Default
        at import is True.
    verbose : bool, optional
        Whether functions print SHAKER's own text output: reports and
        summaries (SASA, bonded, overlap, iterations), `runSim`'s per-stage
        lines, bead neighbours and the mapping report. Warnings and errors are always
        shown. Default at import is True.
    progress : bool, optional
        Whether to show progress bars (mapping, measuring, overlap,
        alignment). Default at import is True.

    Examples
    --------
    For an agent driving SHAKER, keep the text reports but drop figures
    and progress bars from the notebook output::

        shaker.set_options(show=False, progress=False)
    """
    for key, value in (("show", show), ("verbose", verbose), ("progress", progress)):
        if value is not None:
            _OPTIONS[key] = bool(value)


def _resolve(value, key):
    """`value` if given, else the SHAKER-wide default for `key`."""
    return _OPTIONS[key] if value is None else value


def _progress(iterable, **kwargs):
    """A tqdm progress bar, unless turned off with `set_options(progress=False)`."""
    return tqdm(iterable, disable=not _OPTIONS["progress"], **kwargs)
