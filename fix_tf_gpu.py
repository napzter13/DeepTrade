### Notes to Fix CUDA Windows 11:

## Install conda env:
# conda update -n base -c defaults conda
# conda update -n base --all
# conda create --name env3 python=3.6
# conda activate env3

## Install deps:
# conda install -n env3 -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# pip install tensorflow-gpu==2.6
# pip install keras==2.6 --force
# python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

# # http://www.mysmu.edu/faculty/jwwang/post/install-gpu-support-to-tensoflow-on-windows/

