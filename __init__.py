"""
Wan Animate Enhancer Custom Package
Enhanced WanAnimateToVideo with multi-dimensional control
"""

try:
    from .wan_animate_to_video_enhanced import (
        WanAnimateToVideoEnhanced,
        WanAnimateModelEnhancer,
        WanAnimateFFNChunking,
    )
except ImportError:
    # Pytest imports a repository-root __init__.py without package context.
    # ComfyUI's normal package import always uses the relative branch above.
    if __package__:
        raise
    from wan_animate_to_video_enhanced import (
        WanAnimateToVideoEnhanced,
        WanAnimateModelEnhancer,
        WanAnimateFFNChunking,
    )

__version__ = "1.2.1"

NODE_CLASS_MAPPINGS = {
    "WanAnimateToVideoEnhanced": WanAnimateToVideoEnhanced,
    "WanAnimateModelEnhancer": WanAnimateModelEnhancer,
    "WanAnimateFFNChunking": WanAnimateFFNChunking,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanAnimateToVideoEnhanced": "Wan Animate To Video Enhanced (Custom)",
    "WanAnimateModelEnhancer": "Wan Animate Model Enhancer (Custom)",
    "WanAnimateFFNChunking": "WAN FFN Chunking (Custom)",
}

NODE_METADATA = {
    "WanAnimateToVideoEnhanced": {
        "version": "1.2.1",
        "category": "Wan2.2AnimateEnhancer",
        "description": "Enhanced WanAnimateToVideo with motion/expression/pose/background control",
    },
    "WanAnimateModelEnhancer": {
        "version": "1.2.1",
        "category": "Wan2.2AnimateEnhancer",
        "description": "Model enhancer for motion strength control",
    },
    "WanAnimateFFNChunking": {
        "version": "1.2.1",
        "category": "Wan2.2AnimateEnhancer",
        "description": "Branch-local selectable WAN FFN chunking for lower peak VRAM",
    },
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'NODE_METADATA']

print(f"Wan Animate Enhancer Custom v{__version__} loaded - 3 nodes registered")
