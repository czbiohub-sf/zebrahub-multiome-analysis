# pyslingshot_gpu/__init__.py
"""
GPU-accelerated Slingshot trajectory inference

This module provides GPU acceleration for pySlingshot without modifying
the original installation.
"""

# Import main functions
try:
    from .gpu_slingshot import GPUSlingshot, create_slingshot, benchmark_slingshot, check_gpu_availability
    print("✅ GPU Slingshot core functions loaded")
except ImportError as e:
    print(f"⚠️  Error importing GPU Slingshot: {e}")
    # Fallback imports
    try:
        from pyslingshot import Slingshot as GPUSlingshot
        def create_slingshot(data, use_gpu=False, **kwargs):
            return GPUSlingshot(data, **kwargs)
        def benchmark_slingshot(*args, **kwargs):
            print("Benchmark not available without GPU implementation")
        def check_gpu_availability():
            print("GPU check not available without GPU implementation")
            return {'gpu': False}
    except ImportError:
        print("❌ Neither GPU nor original pySlingshot available")

# Import utilities if available
try:
    from .utils import install_requirements, memory_estimate, get_optimal_settings
except ImportError:
    print("⚠️  Utils not available")
    def install_requirements():
        print("Please install: pip install cuml-cu11 cupy-cuda11x pyslingshot")
    def memory_estimate(n_cells):
        return n_cells * 0.00001  # GB
    def get_optimal_settings(n_cells, n_clusters):
        return {'use_gpu': True, 'num_epochs': 10}

__version__ = "1.0.0"

__all__ = [
    'GPUSlingshot',
    'create_slingshot', 
    'benchmark_slingshot',
    'check_gpu_availability',
    'install_requirements',
    'memory_estimate',
    'get_optimal_settings'
]

def get_info():
    """Get GPU Slingshot module information"""
    try:
        gpu_status = check_gpu_availability()
        gpu_available = any(gpu_status.values())
    except:
        gpu_available = False
        
    return {
        'version': __version__,
        'gpu_available': gpu_available,
        'description': 'GPU-accelerated trajectory inference for single-cell data'
    }