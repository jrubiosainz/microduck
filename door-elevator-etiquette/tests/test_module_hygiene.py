#!/usr/bin/env python3
"""Module hygiene: size, dependency isolation, and no unmeasured leftovers.

Three claims that are easy to let rot and expensive to discover late:

* no module outgrows 300 lines of code;
* the headless gate imports no rendering stack at all;
* nothing carried over from the sibling behavior sits here unused, pretending
  to be measured.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
TOOLS = REPO / "tools"
RENDER_MODULES = ("imageio", "imageio.v2", "PIL", "PIL.Image", "matplotlib",
                  "imageio_ffmpeg")


def code_lines(path: Path) -> int:
    """Lines that are neither blank, nor comments, nor docstring bodies."""
    source = path.read_text()
    lines = source.split("\n")
    tree = ast.parse(source)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not (node.body and isinstance(node.body[0], ast.Expr)):
            continue
        if ast.get_docstring(node, clean=False) is None:
            continue
        statement = node.body[0]
        for line in range(statement.lineno,
                          (statement.end_lineno or statement.lineno) + 1):
            docstring_lines.add(line)
    count = 0
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or number in docstring_lines:
            continue
        count += 1
    return count


def python_files():
    return sorted(list(SCRIPTS.glob("*.py")) + list(TOOLS.glob("*.py")))


@pytest.mark.parametrize(
    "path", python_files(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_module_stays_under_300_code_lines(path):
    count = code_lines(path)
    assert count <= 300, f"{path.name} has {count} code lines"


@pytest.mark.parametrize(
    "path", python_files(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_every_module_compiles(path):
    ast.parse(path.read_text())


def test_the_headless_gate_imports_no_rendering_stack():
    """``validate_etiquette.py`` must run with no PIL, imageio or GPU at all.

    Proved by BLOCKING those modules in ``sys.meta_path`` and importing the
    entry point in a clean interpreter, rather than by reading the imports:
    a transitive import through any other module would slip past a grep.
    """
    program = f"""
import sys
BLOCKED = {RENDER_MODULES!r}


class Blocker:
    def find_module(self, name, path=None):
        return self if name.split('.')[0] in {{m.split('.')[0]
                                              for m in BLOCKED}} else None

    def load_module(self, name):
        raise ImportError(f'{{name}} is blocked')

    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {{m.split('.')[0] for m in BLOCKED}}:
            raise ImportError(f'{{name}} is blocked')
        return None


sys.meta_path.insert(0, Blocker())
sys.path.insert(0, {str(SCRIPTS)!r})
import validate_etiquette  # noqa: F401
import rollout_etiquette  # noqa: F401
import etiquette_metrics  # noqa: F401
print('OK')
"""
    result = subprocess.run([sys.executable, "-c", program],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_the_renderer_is_the_only_place_the_rendering_stack_enters():
    """And it must import them INSIDE the render branch, not at module scope."""
    entry = (SCRIPTS / "render_door_lift.py").read_text()
    tree = ast.parse(entry)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([alias.name for alias in node.names]
                     if isinstance(node, ast.Import) else [node.module or ""])
            for name in names:
                assert name.split(".")[0] not in {m.split(".")[0]
                                                  for m in RENDER_MODULES}, name


def test_nothing_from_the_sibling_behavior_is_carried_here_inert():
    """A function the source defines but never calls did not produce evidence.

    The sibling ``lead-me-somewhere`` behavior has an arrival gesture, a
    destination registry and an A* planner; carrying any of them here unused -
    or worse, wiring one in without measuring it on this scene - would be
    shipping unmeasured behavior.

    Checked against IMPORTS AND DEFINED NAMES rather than raw text.  A first
    version grepped for substrings and failed on the word "follower" inside a
    docstring explaining why this behavior does NOT use a follower - which is
    documentation doing its job, not code doing the wrong thing.
    """
    camera = (SCRIPTS / "etiquette_camera.py").read_text()
    assert "INDICATE" not in camera
    assert "self.gesture_yaw" not in camera

    forbidden = {"guide_layout", "guide_states", "guide_machine",
                 "guide_camera", "guide_follower", "guide_planner",
                 "guide_cast", "guide_control", "guide_metrics",
                 "guide_route", "guide_tracker", "guide_aim"}
    for path in list(SCRIPTS.glob("*.py")) + list(TOOLS.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden, (path.name, alias.name)
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "") not in forbidden, (path.name,
                                                              node.module)


def test_every_module_level_function_is_actually_reachable():
    """Nothing defined and never referenced anywhere in the package.

    Catches the specific failure the skill warns about: a helper carried across
    from a sibling behavior that the source defines but never calls, which did
    not produce any of the measured evidence and whose later wiring-in can break
    a gate that previously passed.
    """
    defined: dict[str, str] = {}
    for path in SCRIPTS.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                defined[node.name] = path.name

    corpus = "\n".join(
        path.read_text()
        for path in (list(SCRIPTS.glob("*.py")) + list(TOOLS.glob("*.py"))
                     + list((REPO / "tests").glob("*.py"))))
    orphans = []
    for name, origin in sorted(defined.items()):
        # A definition line plus any other mention means it is referenced.
        if corpus.count(name) <= 1:
            orphans.append(f"{origin}:{name}")
    assert not orphans, f"defined but never referenced: {orphans}"


def test_the_measured_constants_are_not_inherited_from_a_sibling():
    """Every locomotion constant must carry this scene's own measurement."""
    states = (SCRIPTS / "etiquette_states.py").read_text()
    for marker in ("MEASURED", "tools/sweep_commands.py"):
        assert marker in states
    # The turning circle is DERIVED from the sweep, not typed.
    assert "math.radians(18.9)" in states
    assert "math.radians(17.4)" in states


def test_the_thresholds_module_holds_data_only():
    """Judging and measuring live in different files; so do the numbers."""
    tree = ast.parse((SCRIPTS / "etiquette_thresholds.py").read_text())
    for node in tree.body:
        assert isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr,
                                 ast.ImportFrom, ast.Import)), type(node)
