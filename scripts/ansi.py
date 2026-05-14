from __future__ import annotations

import re


ANSI_ESCAPE_RE = re.compile(
    r"""
    \x1B
    (?:
        \[[0-?]*[ -/]*[@-~]
        | \][^\x07]*(?:\x07|\x1B\\)
        | [@-Z\\-_]
    )
    """,
    re.VERBOSE,
)

TERMINAL_GLYPH_TRANSLATION = str.maketrans(
    {
        "─": "-",
        "│": "|",
        "├": "|",
        "└": "`",
        "╭": "+",
        "╮": "+",
        "╰": "+",
        "╯": "+",
        "━": "-",
        "┃": "|",
    }
)


def strip_ansi(text: str) -> str:
    """Remove terminal control sequences and normalize common progress glyphs."""

    text = ANSI_ESCAPE_RE.sub("", text)
    text = text.replace("🎊 ", "")
    return text.translate(TERMINAL_GLYPH_TRANSLATION)
