"""Verify everything about the container image that does not need a container runtime.

The image is built and exercised by the `docker` job in CI. It cannot be built on the
development machine here: Docker Desktop requires administrator rights and so does
`wsl --install`, and WSL reports itself not installed. Rather than leave the whole thing
unverified between pushes, this checks every property of the image that can be
established without a daemon — which is most of them, including the two layers that
actually fail in practice.

What this proves:
  * every dependency resolves to a Linux wheel for the base image's Python, so the
    `pip install` layer needs no compiler;
  * the lock matches requirements.txt, so the image gets the versions the tests ran
    against rather than whatever PyPI serves on build day;
  * no pre-release is pinned;
  * each COPY source exists and the build context is small;
  * the Dockerfile's own RUN steps succeed when run here (demo generation, test suite);
  * the entrypoint's startup checks pass against this checkout;
  * the runtime stage runs as a non-root user and declares a healthcheck.

What it cannot prove: that the base image layers together, that the HEALTHCHECK fires,
or that the process survives a SIGTERM in a real container. Those need a daemon and are
covered by CI.

    .venv/Scripts/python deploy/verify_image.py
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys

# Same directory, and this file is only ever run as a script, so deploy/ is sys.path[0].
# Importing rather than restating the platforms keeps one definition of what the image
# is built for; getting the two out of step is how an arm64 build breaks unnoticed.
import lock_requirements

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKERFILE = os.path.join(ROOT, "Dockerfile")
PYTHON_VERSION = lock_requirements.PYTHON_VERSION

FAILURES: list = []

#: Actions logs need repository admin rights to read through the API, but annotations
#: do not. Emitting each failure as one means a red build can be diagnosed from outside
#: the web UI, which is the difference between reading why it failed and guessing.
ON_GITHUB = os.environ.get("GITHUB_ACTIONS") == "true"


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
        if ON_GITHUB:
            summary = f"{name} — {detail}" if detail else name
            print("::error title=verify_image::" + " ".join(summary.split())[:900])


def report(name: str, detail: str) -> None:
    print(f"  [ -- ] {name} — {detail}")


def dockerfile() -> str:
    with open(DOCKERFILE, encoding="utf-8") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
def check_runtime_available() -> str | None:
    for runtime in ("docker", "podman", "nerdctl"):
        found = subprocess.run(["which", runtime], capture_output=True, text=True,
                               check=False)
        if found.returncode == 0:
            return runtime
    return None


def check_dependencies() -> None:
    """Resolve the lock for every architecture the image can be built on.

    Both, not just this machine's: the image definition is architecture-neutral, and a
    lock that is quietly x86-64 only fails in the pip layer after the whole build
    context has been sent. uv rather than pip because only uv evaluates environment
    markers for the target platform -- see deploy/lock_requirements.py.
    """
    print()
    print("Dependency resolution, per architecture")
    lockfile = os.path.join(ROOT, "requirements.lock.txt")
    for label, target in lock_requirements.TARGETS:
        try:
            # --only-binary :all: is inside compile_for, so a pin with no wheel for
            # this architecture fails here rather than becoming a source build.
            resolved = lock_requirements.compile_for(target, lockfile)
        except RuntimeError as error:
            check(f"every locked version has a wheel for {label}", False,
                  lock_requirements._error_tail(error, 1))
            continue
        check(f"every locked version has a wheel for {label}", True,
              f"{len(resolved)} packages for Python {PYTHON_VERSION}")

    prereleases = [f"{name} {version}" for name, version
                   in sorted(lock_requirements.read_lock().items())
                   if lock_requirements._is_prerelease(version)]
    check("no pre-release is pinned", not prereleases, ", ".join(prereleases) or "none")


def check_lock_is_current() -> None:
    completed = subprocess.run(
        [sys.executable, os.path.join(ROOT, "deploy", "lock_requirements.py"), "--check"],
        capture_output=True, text=True, check=False, cwd=ROOT)
    check("requirements.lock.txt satisfies requirements.txt on both architectures",
          completed.returncode == 0, completed.stdout.strip().splitlines()[-1]
          if completed.stdout.strip() else "")


def check_copy_sources() -> None:
    print()
    print("Build context")
    missing = []
    for line in dockerfile().splitlines():
        stripped = line.strip()
        if not stripped.startswith("COPY ") or "--from=" in stripped:
            continue
        parts = stripped.split()[1:-1]
        for part in parts:
            if part.startswith("--"):
                continue
            if not os.path.exists(os.path.join(ROOT, part)):
                missing.append(part)
    check("every COPY source exists in the build context", not missing,
          ", ".join(missing) if missing else "all present")

    ignored = set()
    ignore_path = os.path.join(ROOT, ".dockerignore")
    if os.path.exists(ignore_path):
        with open(ignore_path, encoding="utf-8") as handle:
            ignored = {line.strip().rstrip("/") for line in handle
                       if line.strip() and not line.startswith("#")}
    check(".dockerignore excludes the data cache and every virtualenv",
          "data" in ignored and any(p.startswith(".venv") and "*" in p for p in ignored),
          f"{len(ignored)} patterns")

    import fnmatch
    total = 0
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if not any(fnmatch.fnmatch(d, p) for p in ignored)]
        for name in files:
            if any(fnmatch.fnmatch(name, p) for p in ignored):
                continue
            with contextlib.suppress(OSError):
                total += os.path.getsize(os.path.join(base, name))
    check("the build context is small enough to send quickly",
          total < 50_000_000, f"{total / 1e6:.1f} MB")


def check_hardening() -> None:
    print()
    print("Runtime hardening")
    text = dockerfile()
    check("the runtime stage drops root", "USER microverse" in text,
          "USER microverse")
    check("a non-root user is created with a fixed uid",
          "useradd" in text and "10001" in text, "uid/gid 10001")
    check("the image declares a healthcheck", "HEALTHCHECK" in text)
    check("the data directory is a volume", 'VOLUME ["/data"]' in text)
    check("the build runs the test suite before shipping",
          "pytest" in text, "RUN python -m pytest")
    check("a multi-stage build keeps tests out of the runtime layer",
          text.count("FROM ") >= 2 and "AS builder" in text,
          f"{text.count('FROM ')} stages")

    entrypoint = os.path.join(ROOT, "deploy", "entrypoint.sh")
    check("the entrypoint exists", os.path.exists(entrypoint))
    if os.path.exists(entrypoint):
        with open(entrypoint, encoding="utf-8") as handle:
            script = handle.read()
        check("the entrypoint validates the data directory before serving",
              "not writable" in script)
        check("the entrypoint execs uvicorn so it becomes PID 1",
              "exec uvicorn" in script)
        check("shutdown is graceful, so an in-flight analysis is not cut off",
              "--timeout-graceful-shutdown" in script)


def check_gate_tools() -> None:
    """The builder must install what its own test gate imports.

    The test tools live in requirements-dev.txt so they stay out of the function
    bundle, and so out of the lock the image installs. `check_build_steps` runs the
    gate in a development environment, which has them, so it passed here while every
    CI build stopped at "No module named pytest". This reads the Dockerfile instead.
    """
    print()
    print("The test gate's own tools")
    text = dockerfile()
    builder = text.split("AS builder", 1)[-1].split("\nFROM ", 1)[0]
    installs = [line for line in builder.splitlines() if "pip install" in line]
    locked = set(lock_requirements.read_lock())
    for tool in ("pytest", "httpx"):
        lines = [line for line in installs if f'"{tool}==' in line or f" {tool}==" in line]
        check(f"the builder installs {tool} for its test gate",
              tool in locked or bool(lines),
              "pinned in the builder" if lines else "missing from the lock and the builder")
        check(f"{tool} stays out of the runtime stage",
              tool not in locked and not any("/install" in line for line in lines),
              "installed outside /install")


def check_build_steps() -> None:
    """Run the Dockerfile's own RUN steps here. This is the closest thing to a build."""
    print()
    print("The Dockerfile's RUN steps, executed in this checkout")
    steps = [
        ("generate the demo datasets",
         [sys.executable, os.path.join(ROOT, "examples", "make_examples.py")]),
        ("the test suite the build gates on",
         [sys.executable, "-m", "pytest", "tests", "-q", "--no-header", "-x"]),
    ]
    for name, command in steps:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                   check=False)
        tail = (completed.stdout.strip().splitlines() or [""])[-1]
        check(name, completed.returncode == 0, tail[:90])


def check_startup_validation() -> None:
    print()
    print("The entrypoint's startup checks, against this checkout")
    probe = (
        "from app.main import app;"
        "from app.core.runner import run_multiverse;"
        "from app.core.evidence import TIER_REPLICATION;"
        "assert app.routes and TIER_REPLICATION;"
        "print(f'{len(app.routes)} routes, {len(TIER_REPLICATION)} validated tiers')"
    )
    completed = subprocess.run([sys.executable, "-c", probe], cwd=ROOT,
                               capture_output=True, text=True, check=False)
    check("the application imports cleanly", completed.returncode == 0,
          completed.stdout.strip() or completed.stderr.strip().splitlines()[-1:][0]
          if completed.stderr.strip() else "")

    examples = [f for f in os.listdir(os.path.join(ROOT, "examples"))
                if f.endswith("_abundance.tsv")]
    check("the demo datasets the landing page needs are present", bool(examples),
          f"{len(examples)} datasets")


def main() -> int:
    print("Container image verification — everything a daemon is not needed for")
    runtime = check_runtime_available()
    if runtime:
        report("a container runtime is available", f"{runtime} — you can also run "
                                                   f"`{runtime} build -t microverse .`")
    else:
        report("no container runtime on this machine",
               "the image is built and exercised by the `docker` job in CI; "
               "installing a runtime here needs administrator rights")

    check_lock_is_current()
    check_dependencies()
    check_copy_sources()
    check_hardening()
    check_gate_tools()
    # Build steps before startup checks, because that is the order the image does it
    # in and because one produces what the other looks for: the demo datasets are
    # generated by examples/make_examples.py and are not in version control, so on a
    # fresh checkout -- every CI run -- checking for them first finds nothing.
    check_build_steps()
    check_startup_validation()

    print()
    if FAILURES:
        print("FAILED: " + ", ".join(FAILURES))
        if ON_GITHUB:
            print("::error title=verify_image::failed checks: " + ", ".join(FAILURES))
        return 1
    print("Every check that does not require a container daemon passed.")
    print("The build itself, and a full analysis driven through the running container,")
    print("are covered by the `docker` job in .github/workflows/ci.yml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
