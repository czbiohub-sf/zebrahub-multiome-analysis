# gpu_slingshot/__init__.py
"""
GPU-accelerated Slingshot trajectory inference

This module provides GPU acceleration for pySlingshot without modifying
the original installation.
"""

from .gpu_slingshot import GPUSlingshot, create_slingshot, benchmark_slingshot
from .utils import check_gpu_availability, install_requirements

__version__ = "1.0.0"

__all__ = [
    'GPUSlingshot',
    'create_slingshot', 
    'benchmark_slingshot',
    'check_gpu_availability',
    'install_requirements'
]

# Auto-check GPU availability on import
try:
    import cupy as cp
    import cuml
    GPU_AVAILABLE = True
    print("✅ GPU Slingshot ready - CuPy and cuML available")
except ImportError as e:
    GPU_AVAILABLE = False
    print(f"⚠️  GPU Slingshot using CPU fallback - {e}")

def get_info():
    """Get GPU Slingshot module information"""
    return {
        'version': __version__,
        'gpu_available': GPU_AVAILABLE,
        'description': 'GPU-accelerated trajectory inference for single-cell data'
    }