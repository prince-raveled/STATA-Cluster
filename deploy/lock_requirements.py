"""Resolve requirements.txt for the container's platform and write requirements.lock.txt.

The Dockerfile installed `requirements.txt` directly, which is a set of lower bounds.
That is fine for a dev checkout and wrong for a shipped image: the versions in the image
are then whatever PyPI served on build day, and the image stops matching what the tests
ran against. It also let a *pre-release* in — `formulaic` declares
`wrapt>=1.17.0rc1; python_version >= "3.13"`, and a specifier containing `rc1` switches
pre-releases on for that project, so a plain resolve picked `wrapt 2.4.1rc1` over the
stable 2.4.0.

This resolves for the image's platform and writes exact pins.

    .venv/Scripts/python deploy/lock_requirements.py             # write the lock
    .venv/Scripts/python deploy/lock_requirements.py --check     # CI gate
    .venv/Scripts/python deploy/lock_requirements.py --outdated  # what has moved on
    .venv/Scripts/python deploy/lock_requirements.py --hold      # retarget, keep versions

Why uv and not pip
------------------
Because pip cannot resolve for a platform it is not running on, and quietly produces a
lock that is wrong when asked to.

`pip --platform` chooses which *wheel tags* are acceptable. It does not change the
environment markers, which still come from the interpreter doing the resolving. So a
lock generated on Windows for a Linux image is resolved with `sys_platform == "win32"`,
and every marker-conditional dependency comes out backwards. That is not hypothetical.
The two lines that decide it are:

    pytest   colorama>=0.4;   sys_platform == "win32"
    uvicorn  uvloop>=0.15.1;  sys_platform != "win32" ... and extra == "standard"

The lock this file used to write carried `colorama`, which is useless in a Linux image,
and omitted `uvloop`, so the container had been running uvicorn on the plain asyncio
loop rather than the one `uvicorn[standard]` asks for, and nothing said so. pip had
answered the question it was asked. It was the wrong question.

uv's `--python-platform` sets the markers as well as the tags, so the answer is the same
whoever runs it. That is the whole reason it is here. It is a build-time tool only: it
is not in requirements.txt, it does not enter the image, and it is needed only to write
or check the lock.

    pip install uv

Architectures
-------------
The image is built on whichever architecture the host has, from one lock, so both are
resolved and both must agree. There is deliberately no second arm64 lock, because a
second lock would be a second set of versions to validate the science against.

The arm64 target is manylinux_2_28 rather than manylinux2014, and the difference is not
cosmetic. kiwisolver, matplotlib and scikit-learn publish manylinux_2_24, _2_26, _2_27
and _2_28 aarch64 wheels and no 2014 one, so resolving against a 2014 floor reports that
all three must be downgraded on arm64. They must not. The floor this implies — glibc
>= 2.28 on arm64 — is cleared by every Debian release the `python:3.12-slim` base image
has been built on (buster 2.28, bullseye 2.31, bookworm 2.36, trixie 2.41).

What `--check` asks
-------------------
It used to re-resolve requirements.txt from scratch and demand the answer be
byte-identical to the lock. That is not a property of this repository — it is a property
of PyPI on the day CI runs, and it stops holding the first time any of the seventy-odd
dependencies cuts a release. So the gate failed while the lock was perfectly good, and
it took the whole docker job down with it.

A lock is correct when it *satisfies* requirements.txt, not when it agrees with the
latest resolve. So `--check` asks the questions that stay true: is the pinned set
exactly the closure of requirements.txt under those pins, on both architectures, using
wheels only, with nothing pre-release. Drift against the newest releases is real
information but it is a decision, not a build failure, so it lives behind `--outdated`.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIREMENTS = os.path.join(ROOT, "requirements.txt")
LOCKFILE = os.path.join(ROOT, "requirements.lock.txt")
CONSTRAINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "constraints.txt")

#: Must match the Dockerfile's base image.
PYTHON_VERSION = "3.12"

#: The architectures the image is built for, as uv target triples. The lock is written
#: from the first and must agree on the second.
TARGETS = (
    # Both at manylinux_2_28. The x86-64 target used to be manylinux2014, which is
    # glibc 2.17 -- far older than the base image, which is Debian and well past 2.28.
    # Asking for wheels that old started excluding real dependencies: `cbor2`, which
    # the Vercel SDK needs, publishes manylinux_2_28 and nothing older, so the
    # resolution failed for a wheel the image could have installed perfectly well.
    ("x86-64", "x86_64-manylinux_2_28"),
    ("arm64", "aarch64-manylinux_2_28"),
)

HEADER = f"""\
# Generated by deploy/lock_requirements.py — do not edit by hand.
#
# Exact versions the container image installs, resolved from requirements.txt for
# Python {PYTHON_VERSION} on Linux. Regenerate after changing requirements.txt:
#
#     python deploy/lock_requirements.py
#
# Every entry resolves to a binary wheel, so the image needs no compiler.
#
# Resolved *for* Linux rather than on it: environment markers decide whether
# uvloop and colorama belong here, and they must be evaluated for the image's
# platform, not for whichever machine ran the generator. One lock covers both
# architectures — every pin below installs unchanged on arm64 (glibc >= 2.28),
# which --check verifies rather than assumes.
"""


def _is_prerelease(version: str) -> bool:
    for marker in ("a", "b", "rc", "dev"):
        head, sep, tail = version.partition(marker)
        if sep and head and head[-1].isdigit() and tail[:1].isdigit():
            return True
    return False


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def uv_binary() -> str:
    """Find uv, or explain how to get it. See the module docstring for why it is used."""
    found = shutil.which("uv")
    if found:
        return found
    try:
        import uv  # noqa: PLC0415 - build-time tool, not an application dependency
        return uv.find_uv_bin()
    except Exception:
        raise SystemExit(
            "uv is needed to resolve the lock and is not installed.\n"
            "    pip install uv\n"
            "pip cannot be used instead: its --platform changes wheel tags but not "
            "environment markers, so it resolves a Linux image's lock with the host's "
            "markers. See the comment at the top of deploy/lock_requirements.py."
        ) from None


def _relative(path: str) -> str:
    """A path relative to ROOT, or the path itself if it lies outside.

    Everything is passed to uv relative to the repository root, and uv is run with
    ROOT as its working directory, because uv's -c does not survive a space in an
    absolute path: given
    `-c "D:/Prince/Projects/Web-techP4 idea/.../constraints.txt"` it reports
    `File not found: D:/Prince/Projects/Web-techP4`, having stopped at the space.
    The positional requirements argument handles the same path correctly, so this
    is specific to the constraint flag. Relative paths have no spaces here and
    sidestep it entirely.
    """
    try:
        return os.path.relpath(path, ROOT).replace(os.sep, "/")
    except ValueError:
        return path


def compile_for(target: str, requirements: str, constraints: tuple = ()) -> dict:
    """Resolve `requirements` for one target. Returns {normalised name: version}."""
    with tempfile.TemporaryDirectory() as workdir:
        output = os.path.join(workdir, "out.txt")
        command = [
            uv_binary(), "pip", "compile", _relative(requirements),
            "--python-platform", target,
            "--python-version", PYTHON_VERSION,
            "--only-binary", ":all:",
            "--no-header", "--no-annotate",
            "--quiet",
            "-o", output,
        ]
        for constraint in constraints:
            if os.path.exists(constraint):
                command += ["-c", _relative(constraint)]
        completed = subprocess.run(command, capture_output=True, text=True,
                                   check=False, cwd=ROOT)
        if completed.returncode != 0:
            raise RuntimeError((completed.stdout + completed.stderr).strip())
        with open(output, encoding="utf-8") as handle:
            text = handle.read()
    return parse_pins(text)


def parse_pins(text: str) -> dict:
    pins = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if line and "==" in line:
            name, version = line.split("==", 1)
            pins[_normalise(name)] = version.split(";")[0].strip()
    return pins


def read_lock() -> dict:
    if not os.path.exists(LOCKFILE):
        return {}
    with open(LOCKFILE, encoding="utf-8") as handle:
        return parse_pins(handle.read())


def render(pins: dict) -> str:
    lines = [HEADER]
    lines += [f"{name}=={version}" for name, version in sorted(pins.items())]
    return "\n".join(lines) + "\n"


def _error_tail(error: Exception, lines: int = 3) -> str:
    kept = [line for line in str(error).splitlines() if line.strip()][-lines:]
    return " / ".join(kept)


def check_architecture(label: str, target: str, pins: dict) -> list:
    """Resolve requirements.txt *under the lock* and confirm the lock is the answer.

    Constraining the resolve with the lock itself is what makes this deterministic: uv
    is asked whether these exact pins can satisfy requirements.txt on this target,
    which is the property the image depends on, rather than what the index happens to
    offer today.
    """
    try:
        resolved = compile_for(target, REQUIREMENTS, (LOCKFILE, CONSTRAINTS))
    except RuntimeError as error:
        return [f"{label}: the lock cannot satisfy requirements.txt — "
                f"{_error_tail(error)}"]

    problems = []
    for name in sorted(set(pins) | set(resolved)):
        want, have = pins.get(name), resolved.get(name)
        if want is None:
            problems.append(f"{label}: {name} {have} is needed but is not in the lock")
        elif have is None:
            problems.append(f"{label}: {name} {want} is locked but nothing requires it")
        elif want != have:
            problems.append(f"{label}: {name} resolves to {have}, not the locked {want}")
    return problems


def run_check(annotate: bool = False) -> int:
    pins = read_lock()
    if not pins:
        print(f"{os.path.relpath(LOCKFILE, ROOT)} does not exist — run "
              "python deploy/lock_requirements.py")
        return 1

    problems = [f"{name} {version} is a pre-release"
                for name, version in sorted(pins.items()) if _is_prerelease(version)]
    for label, target in TARGETS:
        problems += check_architecture(label, target, pins)

    if problems:
        print(f"{os.path.relpath(LOCKFILE, ROOT)} does not satisfy requirements.txt:")
        for problem in problems:
            print(f"    {problem}")
            if annotate:
                print("::error title=lock::" + " ".join(problem.split())[:900])
        print("\nRun python deploy/lock_requirements.py --hold to retarget it without "
              "moving any\nversion, or without --hold to take the newest and revalidate "
              "the science against them.")
        return 1

    print(f"{os.path.relpath(LOCKFILE, ROOT)} satisfies requirements.txt "
          f"({len(pins)} packages, all wheels, no pre-releases, "
          f"{' and '.join(label for label, _ in TARGETS)}).")
    return 0


def run_outdated() -> int:
    """Report drift against the newest releases. Informational: never fails."""
    pins = read_lock()
    latest = compile_for(TARGETS[0][1], REQUIREMENTS, (CONSTRAINTS,))
    moved = sorted(name for name in set(pins) | set(latest)
                   if pins.get(name) != latest.get(name))
    if not moved:
        print(f"{os.path.relpath(LOCKFILE, ROOT)} is what a fresh resolve would pick.")
        return 0
    print(f"{len(moved)} package(s) have moved since the lock was written. The lock is "
          f"still valid;\nrebuilding it means revalidating the science against the new "
          f"versions.\n")
    for name in moved:
        print(f"    {name:24s} locked {pins.get(name) or '(absent)':<16s} "
              f"latest {latest.get(name) or '(absent)'}")
    return 0


def run_write(hold: bool) -> int:
    """Resolve and write the lock. With `hold`, keep every already-locked version.

    `--hold` exists for retargeting rather than upgrading. When the resolution itself
    was wrong — as it was while the lock was generated with the host's environment
    markers instead of the image's — the fix is to re-resolve for the right target
    while leaving every version alone, so the correction can be reviewed on its own
    rather than buried in a dependency bump.
    """
    constraints = (CONSTRAINTS, LOCKFILE) if hold else (CONSTRAINTS,)
    try:
        pins = compile_for(TARGETS[0][1], REQUIREMENTS, constraints)
    except RuntimeError as error:
        sys.stderr.write(str(error) + "\n")
        return 1

    prereleases = [(n, v) for n, v in sorted(pins.items()) if _is_prerelease(v)]
    if prereleases:
        print("Refusing to write a lock containing pre-releases:")
        for name, version in prereleases:
            print(f"    {name} {version}")
        print(f"\nAdd a stable bound to {os.path.relpath(CONSTRAINTS, ROOT)} and rerun.")
        return 1

    # The other architectures must reach the same answer, or there is no single lock.
    for label, target in TARGETS[1:]:
        try:
            other = compile_for(target, REQUIREMENTS, constraints)
        except RuntimeError as error:
            print(f"Refusing to write a lock {label} cannot satisfy:\n"
                  f"    {_error_tail(error)}")
            return 1
        differences = [f"{n}: {pins.get(n) or '(absent)'} on {TARGETS[0][0]}, "
                       f"{other.get(n) or '(absent)'} on {label}"
                       for n in sorted(set(pins) | set(other))
                       if pins.get(n) != other.get(n)]
        if differences:
            print("Refusing to write a lock the architectures disagree on:")
            for difference in differences:
                print(f"    {difference}")
            return 1

    with open(LOCKFILE, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(render(pins))
    print(f"wrote {os.path.relpath(LOCKFILE, ROOT)} — {len(pins)} packages, all wheels, "
          f"no pre-releases, {' and '.join(label for label, _ in TARGETS)}"
          + (", versions held" if hold else ""))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true",
                       help="fail if the lock is missing, or no longer satisfies "
                            "requirements.txt on either architecture")
    group.add_argument("--outdated", action="store_true",
                       help="report what a fresh resolve would pick instead")
    group.add_argument("--hold", action="store_true",
                       help="rewrite the lock without moving any locked version")
    args = parser.parse_args()

    if args.check:
        return run_check(annotate=os.environ.get("GITHUB_ACTIONS") == "true")
    if args.outdated:
        return run_outdated()
    return run_write(hold=args.hold)


if __name__ == "__main__":
    sys.exit(main())
