import pandas as pd
import numpy as np
import scipy as sp
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import warnings
from .utilities import *
from . Echo_experiment import *
from .Echo_trace import *
from .fitting_tools import *
from .Sweep_experiment import *
update_matplot_style()
et_colors = color_palette()


class T1_measurement(Sweep_experiment):

    '''
    For experiments measuring T1. Simple add ons to Sweep_experiment for fitting
    '''

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        rep_time = kwargs.get('rep_time',None) #period of looped wait time in pulse sequence
        if rep_time:
            self.convert_reps_to_time(rep_time)

    def fit_integrated_echos(self,signals=['I','Q','IQ'],plot=True,**kwargs):
        '''
        Use T1_fits class to fit the signals specified

        guesses : work as p0 in sp.optimize.curve_fit utilized in T1_fits. Should be given as (T1,a,b)
        '''

        guesses = kwargs.get('guesses',{i : None for i in signals})
        self.T1_fits = {i : T1_fit(self.integrated_echos[i],guess=guesses[i]) for i in signals}

        if plot:
            self.plot_fit_integrated_echos(signals,**kwargs)

    def plot_fit_integrated_echos(self,signals,save_name=None,**kwargs):

        legend_loc = kwargs.get('legend_loc', {i: 1 for i in signals})
        axes = kwargs.get('axes',None)
        _flag_axes_supplied = True
        if not axes:
            fig,axes = generate_axes(shape=(3,1))
            _flag_axes_supplied = False

        axes = self.plot_integrated_echos(axes=axes)
        for i in zip(axes,['I','Q','IQ']):
            if i[1] in signals:
                i[0].plot(self.T1_fits[i[1]].x,self.T1_fits[i[1]].fit)
                i[0].add_artist(AnchoredText(self.T1_fits[i[1]].result_string(),loc=legend_loc[i[1]]))

        if _flag_axes_supplied:
            return(axes)
        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        else:
            plt.show()

    def convert_reps_to_time(self,fixed_delay,rep_time):
        '''
        The wait time in T1 experiments (self.columns) is often saved as the number of times some small period of time
        is repeated. This converts the number of repetitions into a time in ms and relabels the columns

        fixed_delay : us
        rep_time : us
        '''

        self.relabel_columns(rep_time*self.columns*1e-3 + fixed_delay*1e-3)
        self.sweep_parameter = 'Wait Time (ms)'