
import glob
import numpy as np
import pandas as pd
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import seaborn as sns
from SALib.sample import saltelli

problem = {
    'num_vars': 2,
    'names': ['dps', 'eip', 'pvtoh'],
    'bounds': [
        [0.0, 1.0],
        [0, 20],
        [0.0, 1.0]
    ]
}
params = saltelli.sample(problem, 16)
print(params)
