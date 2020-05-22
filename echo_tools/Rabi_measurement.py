import pandas as pd
import numpy as np
import scipy as sp
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import warnings
from .utilities import *
from .Echo_trace import *
from .fitting_tools import *
from .Sweep_experiment import *
update_matplot_style()
et_colors = color_palette()


class Rabi_measurement(Sweep_experiment):

    '''
    For experiments measuring Rabis by varying pulse amplitude.
    '''

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.sweep_parameter = r'$\pi$ Pulse Power (dBm)'