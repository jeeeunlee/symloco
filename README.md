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
```bash
conda create -n "symloco" python=3.11 -y
conda activate symloco -y
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install stable-baselines3[extra]
pip install gymnasium[mujoco]
conda install -y -c conda-forge tensorboard
conda install -y -c conda-forge scipy
```

## Dependencies for GPU: Brax, Mujoco Playground, and others

Note: we install brax from Github (instead of from pip) because easily saving checkpoint files is currently an unreleased feature.

```bash
pip install -Uq "jax[cuda]"
pip install mujoco_mjx
cd dependencies
git clone https://github.com/google/brax.git
cd brax
pip install -e
cd ../../
pip install playground
pip install tqdm
```

## Training and Testing

The scripts for training/testing can be found in `src/tests`.

### Training

Start a training run using the following command:
```bash
python src/tests/main_<robot>.py train -n <model_name> -s
```

For training, the full list of arguments is:
```bash
--model_name (-n): name of the model (required)
--n_envs (-e): number of environments to train with (default is 16)
--use_sym_policy (-s): whether to use the symmetric policy (default is false)
--velocity_profile (-v): "oneway" or "bothway" (default is "oneway")
```

For training on GPU, the argument list is:
```bash
--run_name (-n): name of the run to train (required)
--use_sym_policy (-s): whether to use the symmetric policy (default is false)
```

### Testing

Test a model using the following command (on CPU):
```bash
python src/tests/main_<robot>.py test --model-path models/<model_name>/<model_file>.zip
```

For testing, the full list of arguments is:
```bash
--model_path (-mp): path of model to test (required)
--n_envs (-e): number of environments to test with (default is 16)
--velocity_profile (-v): "oneway" or "bothway" (default is "oneway")
```

For testing on GPU, the argument list is:
```bash
--run_name (-n): name of the run to test (required)
--checkpoint_name (-c): name of checkpoint folder in models/<run_name> (required)
```

For example:
```bash
python src/tests/main_<robot>.py test -n playground_run -c 000022282240
```