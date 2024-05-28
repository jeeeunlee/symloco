# SYMLOCO
Reinforcement Learning for SYMmetrical LOCOmotion 
------------
## Install

### Python Virtual Environment Setup
Create the right version of python virtual environmnet corresponding to CUDA (e.g., CUDA 12.1 works well on python 3.11)
If gpu exists and cuda installed check cuda version by 
```
nvcc --version
```

cuda 12.1 example:
> nvcc: NVIDIA (R) Cuda compiler driver
> Copyright (c) 2005-2023 NVIDIA Corporation
> Built on Mon_Apr__3_17:16:06_PDT_2023
> Cuda compilation tools, release 12.1, V12.1.105
> Build cuda_12.1.r12.1/compiler.32688072_0

Then, you can create python virtual env and install pytorch, stable-baseline3, gymnasium, etc.
```
conda create -n "symloco" python=3.11 -y
conda activate symloco -y
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install stable-baselines3[extra]
pip install gymnasium[mujoco]
conda install -y -c conda-forge tensorboard
conda install -y -c conda-forge scipy
```
