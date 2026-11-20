"""Proceed-or-terminate questions asked on the terminal during a pipeline run.

Used when a run can go on only at a cost the user should accept: a camera with no usable
trace (``Trace/traceValidation.confirm_unusable_master_traces``), or too little disk space
(``Utils/diskSpace.disk_space_gate``). The question goes to the real terminal
(``sys.__stdout__``), because during a curated run ``sys.stdout`` is redirected to the log.
"""
import sys


class NoTerminalError(RuntimeError):
    """Nobody can answer: standard input is not a terminal (a batch or background run)."""


def ask_proceed_or_terminate(question, interactive=None, input_fn=input, stream=None) -> bool:
    """Ask ``question`` until the answer is proceed or terminate.

    Args:
        question: the question, without the ``[p/t]`` choice.
        interactive: whether someone can answer; defaults to ``sys.stdin.isatty()``.
        input_fn: reads the answer (for tests).
        stream: where the question is written; defaults to ``sys.__stdout__``.

    Returns:
        True to proceed, False to terminate. End of input counts as terminate.

    Raises:
        NoTerminalError: not interactive.
    """
    out = stream or sys.__stdout__
    if interactive is None:
        interactive = sys.stdin is not None and sys.stdin.isatty()
    if not interactive:
        raise NoTerminalError(question)
    while True:
        out.write(f"{question} [p/t]: ")
        out.flush()
        try:
            answer = input_fn().strip().lower()
        except EOFError:
            answer = 't'
        if answer in ('t', 'terminate'):
            return False
        if answer in ('p', 'proceed'):
            return True
        out.write("Please answer 'p' to proceed or 't' to terminate.\n")
