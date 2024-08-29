import tensorflow as tf
import os
import subprocess

print("TensorFlow version:", tf.__version__)
print("CUDA available:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("CUDA_PATH:", os.environ.get('CUDA_PATH', 'Not set'))
print("CUDA_HOME:", os.environ.get('CUDA_HOME', 'Not set'))

if tf.test.is_built_with_cuda():
    print("CUDA version:", tf.sysconfig.get_build_info()["cuda_version"])
    print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])

# Выполнение nvcc --version
try:
    nvcc_output = subprocess.check_output(["nvcc", "--version"]).decode()
    print("nvcc version:\n", nvcc_output)
except Exception as e:
    print("Error running nvcc:", str(e))

# Выполнение nvidia-smi
try:
    nvidia_smi_output = subprocess.check_output(["nvidia-smi"]).decode()
    print("nvidia-smi output:\n", nvidia_smi_output)
except Exception as e:
    print("Error running nvidia-smi:", str(e))

# Простой тест GPU
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print("GPU computation result:", c)
except RuntimeError as e:
    print("GPU test failed:", str(e))

# Проверка путей CUDA в PATH
path = os.environ.get('PATH', '')
cuda_paths = [p for p in path.split(os.pathsep) if 'CUDA' in p]
print("CUDA paths in PATH:")
for p in cuda_paths:
    print(p)