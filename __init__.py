"""Ultimate UV Artist - automatic artist-style UV unwrapping for ComfyUI (MIT)."""
__version__ = "1.0.0"


async def comfy_entrypoint():
    from comfy_api.latest import ComfyExtension
    from .nodes import NODES

    class UltimateUVArtistExtension(ComfyExtension):
        async def get_node_list(self):
            return list(NODES)

    return UltimateUVArtistExtension()
