import subprocess
import sys
import os
import tensorflow as tf

def run_command(cmd):
    """
    Runs a shell command safely, printing output in real time.
    """
    print(f"\n[Running]: {cmd}\n")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(process.stdout.readline, b''):
        print(line.decode().rstrip())
    process.stdout.close()
    process.wait()

def install_or_upgrade_tensorflow():
    """
    Upgrades pip, uninstalls any old TF, and installs the latest TF.
    """
    # 1) Upgrade pip to the latest
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    # 2) Uninstall existing tensorflow if present
    run_command(f"{sys.executable} -m pip uninstall -y tensorflow tensorflow-gpu tensorflow-intel")

    # 3) Install the latest stable tensorflow (which has GPU support on Windows)
    run_command(f"{sys.executable} -m pip install --upgrade tensorflow")

def check_tensorflow_gpu():
    """
    Imports tensorflow and runs checks to confirm GPU availability and usage.
    """
    import tensorflow as tf

    print("\n--- Checking TensorFlow version ---")
    print("TensorFlow version:", tf.__version__)

    # Check if TensorFlow sees any GPUs
    print("\n--- Checking for GPU devices ---")
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("No GPU was found by TensorFlow!")
        print("If you expect a GPU, ensure you have a compatible CUDA-compatible driver installed.\n")
    else:
        print(f"Number of GPUs available to TensorFlow: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i} details: {gpu}")

    # Enable device placement logging to see if GPU is actually used
    tf.debugging.set_log_device_placement(True)

    print("\n--- Testing a quick GPU operation (matrix multiplication) ---")
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    print("Matrix multiplication result shape:", c.shape)

    print("\n--- Testing a small Keras model training ---")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1000,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Dummy data
    X = tf.random.normal([2000, 1000])
    y = tf.random.uniform([2000], minval=0, maxval=10, dtype=tf.int32)

    model.fit(X, y, epochs=1, batch_size=128)

def main():
    print("===== TensorFlow GPU Fix & Test Script =====")
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices('GPU'))
    print("--------------------------------------------")

    # 1) Attempt to fix environment by installing/ upgrading TF
    install_or_upgrade_tensorflow()

    # 2) Now do the check to see if it works
    try:
        check_tensorflow_gpu()
    except ImportError as e:
        print("\n[Error] Could not import TensorFlow after installation:", e)
    except Exception as ex:
        print("\n[Error] Some error occurred during the test:", ex)

    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices('GPU'))
    print("\n===== Script finished! =====")

if __name__ == "__main__":
    main()
