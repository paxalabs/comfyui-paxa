from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from comfyui_paxa import __version__, client, nodes

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
WORKFLOWS = sorted((ROOT / "example_workflows").glob("*.json"))
# Core nodes the examples may use, all in ComfyUI since 0.3.40. Note is the
# frontend's own sticky note.
CORE_NODES = {"SaveAudioMP3": ["filename_prefix", "quality"], "PreviewAny": [], "Note": ["text"]}


def test_version_matches_pyproject():
    assert PYPROJECT["project"]["version"] == __version__
    assert client.CLIENT_TAG == f"comfyui-paxa/{__version__}"


def test_requirements_match_pyproject():
    lines = (ROOT / "requirements.txt").read_text().split()
    assert lines == PYPROJECT["project"]["dependencies"]
    assert not any(name.startswith(("torch", "numpy")) for name in lines)


def test_registry_metadata():
    name = PYPROJECT["project"]["name"]
    # The registry's rules for a node id (docs.comfy.org/registry/specifications).
    assert len(name) < 100
    assert re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*", name)
    comfy = PYPROJECT["tool"]["comfy"]
    assert comfy["PublisherId"] == "paxalabs"
    assert comfy["DisplayName"]
    assert comfy["Icon"] == "https://raw.githubusercontent.com/paxalabs/comfyui-paxa/main/icon.svg"
    assert (ROOT / "icon.svg").read_text().startswith("<svg")
    assert PYPROJECT["project"]["license"] == {"file": "LICENSE"}
    assert (ROOT / "LICENSE").read_text().startswith("MIT License")
    assert PYPROJECT["project"]["urls"]["Repository"] == "https://github.com/paxalabs/comfyui-paxa"


def test_key_file_is_never_committed_or_published():
    assert "paxa_api_key.txt" in (ROOT / ".gitignore").read_text().split()
    ignored = (ROOT / ".comfyignore").read_text().split()
    assert {"tests/", "scripts/", ".github/"} <= set(ignored)


def test_no_key_shaped_strings_anywhere():
    key_shaped = re.compile(r"pxa_[A-Za-z0-9]{6,}")
    skip = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
    for path in ROOT.rglob("*"):
        if path.is_file() and not skip & set(path.relative_to(ROOT).parts):
            text = path.read_bytes().decode("utf-8", errors="ignore")
            assert not key_shaped.search(text), path


def test_readme_names_the_v1_only_voices_in_list_order():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    listed = ", ".join(f"`{voice}`" for voice in nodes.V1_ONLY[:-1])
    assert f"{listed} and `{nodes.V1_ONLY[-1]}`" in readme


def test_voice_snapshot_has_the_contract_fields(voices):
    fields = {
        "id",
        "name",
        "gender",
        "language",
        "accent",
        "experimental",
        "emotion_tags",
        "description",
        "model",
    }
    assert voices
    for voice in voices:
        assert set(voice) == fields
        assert voice["model"] in client.MODELS


# Example workflows: what the template browser offers.


def widget_names(node_type: str) -> list[str]:
    if node_type in CORE_NODES:
        return CORE_NODES[node_type]
    node_class = nodes.NODE_CLASS_MAPPINGS[node_type]
    names = []
    for section in ("required", "optional"):
        for name, spec in node_class.INPUT_TYPES().get(section, {}).items():
            options = spec[1] if len(spec) > 1 else {}
            if isinstance(spec[0], list) or (
                spec[0] in ("STRING", "INT", "FLOAT", "BOOLEAN") and not options.get("forceInput")
            ):
                names.append(name)
    return names


def socket_types(node_type: str, direction: str) -> list[str]:
    node_class = nodes.NODE_CLASS_MAPPINGS[node_type]
    if direction == "outputs":
        return list(node_class.RETURN_TYPES)
    types = []
    for section in ("required", "optional"):
        for spec in node_class.INPUT_TYPES().get(section, {}).values():
            options = spec[1] if len(spec) > 1 else {}
            if spec[0] == "AUDIO" or options.get("forceInput"):
                types.append(spec[0])
    return types


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_example_workflow(path):
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert workflow["version"] == 0.4
    by_id = {node["id"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(by_id)
    for node in workflow["nodes"]:
        assert node["type"] in nodes.NODE_CLASS_MAPPINGS or node["type"] in CORE_NODES
        assert len(node["widgets_values"]) == len(widget_names(node["type"])), node["type"]
        if node["type"] in nodes.NODE_CLASS_MAPPINGS:
            assert node["properties"]["cnr_id"] == PYPROJECT["project"]["name"]
            assert node["properties"]["ver"] == __version__
            specs = dict(
                nodes.NODE_CLASS_MAPPINGS[node["type"]].INPUT_TYPES()["required"],
                **nodes.NODE_CLASS_MAPPINGS[node["type"]].INPUT_TYPES().get("optional", {}),
            )
            for name, value in zip(widget_names(node["type"]), node["widgets_values"], strict=True):
                if isinstance(specs[name][0], list):
                    assert value in specs[name][0], (node["type"], name, value)
    for link_id, source, source_slot, target, target_slot, kind in workflow["links"]:
        out = by_id[source]["outputs"][source_slot]
        into = by_id[target]["inputs"][target_slot]
        assert link_id in out["links"] and into["link"] == link_id
        assert out["type"] == kind and into["type"] in (kind, "*")
        if by_id[source]["type"] in nodes.NODE_CLASS_MAPPINGS:
            assert socket_types(by_id[source]["type"], "outputs")[source_slot] == kind
        if by_id[target]["type"] in nodes.NODE_CLASS_MAPPINGS:
            assert socket_types(by_id[target]["type"], "inputs")[target_slot] == kind
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])


def test_there_are_example_workflows():
    assert [path.name for path in WORKFLOWS] == [
        "Paxa Text to Speech.json",
        "Paxa Voiceover with Subtitles.json",
    ]
