import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from sampling_example import heuristic_sampling

"""
We will make a simple NN that will be trained to classify points on if they belong to circle 0 or 1.
Each circle is a unit circle with radius of 1 with one centered at (-1, 0) and the other centered at (1, 0). 
This a nonlinear classification problems since if the points lies in a certain circles requires computing a nonlinear 
expression. since the two circles near each other on one side, this should allow us to see the impact of broadening the distributions.
In order to simulate the behavoir we want we will have a seed dataset that is sampled from gaussians that are more tightly clustered
around the circle centers while the test will be sampled from a uniform distribtion over the circles. 
"""


