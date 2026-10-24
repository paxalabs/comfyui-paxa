"""Paxa Labs nodes for ComfyUI: text to speech and speech to text."""

# ComfyUI imports this folder as a package. pytest also imports this file,
# alone, because it sits above the tests; there it has nothing to load.
if __package__:
    from .comfyui_paxa.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
