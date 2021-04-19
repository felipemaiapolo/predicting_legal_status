#%env PYTHONHASHSEED=0

import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']=""
os.environ['TF_CUDNN_USE_AUTOTUNE'] ='0'

if os.environ.get("PYTHONHASHSEED") != "0":
    raise Exception("You must set PYTHONHASHSEED=0 when starting the Jupyter server to get reproducible results.")

random_seed=22

import numpy as np
import random
import tensorflow as tf

np.random.seed(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)