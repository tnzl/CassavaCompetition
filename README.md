# PyTXLA
PyTXLA is a frame work built on top of PyTorch-XLA to simplify training PyTorch models on cloud TPUs.

## Features
1. Callbacks : `callbacks` lets you run sub tasks letting user to tweak the training process, models and create custom callbacks.
2. Data module : `data` contains functions and classes for distributing data over TPU.
3. Learner : `learner` is a module that lets you train PyTorch models without worrying about TPUs.
4. Setup : `setup` installs the requirements and helps you easily ship you experiments over other systems.
