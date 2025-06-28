import tensorflow as tf
import numpy as np
import dask
import matplotlib
import seaborn as sns
import jupyter
import platform

print("="*40)
print("🔧 Environment Info")
print("="*40)
print(f"Python Version     : {platform.python_version()}")
print(f"Platform           : {platform.system()} {platform.release()}")
print(f"TensorFlow Version : {tf.__version__}")
print(f"Num GPUs Available : {len(tf.config.list_physical_devices('GPU'))}")
print(f"TensorFlow Devices : {tf.config.list_physical_devices()}")

print("\n📦 Package Versions")
print("="*40)
print(f"NumPy Version      : {np.__version__}")
print(f"MatplotLib Version : {matplotlib.__version__}")
print(f"Seaborn Version    : {sns.__version__}")
print(f"Dask Version       : {dask.__version__}")

try:
    import geoip2
    print(f"GeoIP2 Version     : {geoip2.__version__}")
except ImportError:
    print("GeoIP2 Version     : Not installed")

print("\n✅ TensorFlow is using GPU!" if tf.config.list_physical_devices('GPU') else "❌ TensorFlow is using CPU only.")
