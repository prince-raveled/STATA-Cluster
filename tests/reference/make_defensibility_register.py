"""Write docs/defensibility_register.json from the engine's own register (plan §26.3).

Every pruning rule (v2's R1-R7 and v3's R8-R9) and every fork carries a reason, its
citations and a Del Giudice & Gangestad (2021) type in `app/core/validity.py`
(`DEFENSIBILITY`). The /about page renders that same structure; this writes it out as
the record a reviewer can read without running anything. `tests/test_defensibility.py`
fails if the file on disk differs from what this would write now.

    python tests/reference/make_defensibility_register.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from app.core.validity import defensibility_register  # noqa: E402

OUTPUT = os.path.join(ROOT, "docs", "defensibility_register.json")


def render() -> str:
    """The file's exact text. No timestamp, so an unchanged register is an unchanged
    file and the drift test can compare bytes."""
    return json.dumps(defensibility_register(), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    text = render()
    with open(OUTPUT, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    register = defensibility_register()
    print(f"wrote {os.path.relpath(OUTPUT, ROOT)}: {len(register['rules'])} rules, "
          f"{len(register['forks'])} forks, {len(register['citations'])} citations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
