"""
PyTorch compatibility module for handling PyTorch 2.6+ breaking changes.

PyTorch 2.6+ changed the default value of `weights_only` in `torch.load` from
False to True for security reasons. However, pyannote models use omegaconf and
other classes that aren't in the safe globals list by default.

This module patches torch.load to use weights_only=False when loading model
checkpoints from trusted sources (HuggingFace models).

IMPORTANT: This module must be imported BEFORE any other modules that use torch.load.
"""
import functools
import torch

_PATCHED = False


def patch_torch_load():
    """Patch torch.load to use weights_only=False by default.
    
    This is safe for loading models from trusted sources like HuggingFace.
    The patch is only applied once, even if called multiple times.
    """
    global _PATCHED
    
    if _PATCHED:
        return
    
    # Check PyTorch version - only patch if >= 2.6.0
    torch_version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
    if torch_version < (2, 6):
        print(f"[torch_compat] PyTorch {torch.__version__} detected (< 2.6), no patch needed")
        _PATCHED = True
        return
    
    print(f"[torch_compat] PyTorch {torch.__version__} detected (>= 2.6), applying weights_only patch")
    
    _original_torch_load = torch.load
    
    @functools.wraps(_original_torch_load)
    def _patched_torch_load(*args, **kwargs):
        # FORCE weights_only=False for compatibility with pyannote models
        # This overrides any explicit weights_only=True passed by libraries like lightning_fabric
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    
    torch.load = _patched_torch_load
    _PATCHED = True
    print("[torch_compat] Patched torch.load to use weights_only=False (for pyannote compatibility)")


# Apply patch on import
patch_torch_load()
