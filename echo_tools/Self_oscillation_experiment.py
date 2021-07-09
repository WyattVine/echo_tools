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


class Self_oscillation_experiment(Sweep_experiment):

    '''
    For experiments studying parametric self-oscillations. Adds functions related to thresholding the data
    and finding transitions between quiet and self-oscillating resonator states.
    '''

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.sweep_parameter = 'Shot'
        self.transitions = {}

    @property
    def nshots(self):
        return self.Is.shape[1]

    def remove_baseline_method2(self):
        '''an additional method to remove baseline. Required for self-oscillation experiments because acquisition might be stopped while
        resonator is still self-oscillating, so cannot remove baseline with 'echo' centered'''
        self.Is -= self.Is.loc[self.noise_range[0]:self.noise_range[-1],:].mean().mean()
        self.Qs -= self.Qs.loc[self.noise_range[0]:self.noise_range[-1],:].mean().mean()

    def create_IQhist(self,type='magnitude',bound=1,nbins=1000,plot=False,**kwargs):
        '''creates histogram of I and Q or |IQ|

        type: 'magnitude' or 'complex' -- specifies histrogram of |IQ| or I,Q
        bound: upper range of histogram bins
        nbins: number of bins in histogram
        '''

        if type == 'complex':
            df = self.Is + 1j*self.Qs
            IQhist = pd.DataFrame(columns=range(df.shape[1]),index=range(nbins),dtype=np.complex64)
            for i in range(df.shape[1]):
                Ihist,bin_edges = np.histogram(df.iloc[:,i].apply(np.real),bins=nbins,range=[-bound,bound])
                Qhist,bin_edges = np.histogram(df.iloc[:,i].apply(np.imag),bins=nbins,range=[-bound,bound])
                IQhist.at[:,i] = Ihist + 1j*Qhist
        elif type == 'magnitude':
            df = self.IQs
            IQhist = pd.DataFrame(columns=range(df.shape[1]),index=range(nbins),dtype=np.float64)
            for i in range(df.shape[1]):
                hist,bin_edges = np.histogram(df.iloc[:,i].apply(np.abs),bins=nbins,range=[0,bound])
                IQhist.at[:,i] = hist
        IQhist.index = [0.5*(bin_edges[i]+bin_edges[i+1]) for i in range(len(bin_edges)-1)]
        self.IQhist = IQhist.sum(axis=1)
        if plot:
            self.plot_IQhist(**kwargs)

    def plot_IQhist(self,save_name=None,**kwargs):

        axes = kwargs.get('axes', None)
        _flag_axes_supplied = True
        if not axes:
            fig, axes = generate_axes(shape=(1,1))
            _flag_axes_supplied = False

        axes[0].plot(self.IQhist.index,self.IQhist)
        axes[0].set_xlabel('|IQ|')
        axes[0].set_ylabel('Count')
        if _flag_axes_supplied:
            return(axes)
        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        else:
            plt.show()

    def IQ_threshold_assign(self,threshold):
        '''
        produces a mapping of self.IQs that corresponds to whether |IQ| is greater than or less than a threshold value
        '''
        self.IQ_threshold = threshold
        self.IQ_th = self.IQs.apply(np.abs).applymap(lambda x: 1 if x >= threshold else 0)
        self.find_transitions()

    def find_transitions(self):
        '''
        Shifts a thresholded dataframe by 1 step to identify transitions between quiet and self-oscillating states.
        df: a thresholded dataframe. i.e. a series of traces that have been processed via threshold_assign()'''
        diff = self.IQ_th.diff().iloc[1:,:]
        self.transitions['up_full'] = diff.where(diff==1,0) #full record of transitions from quiet to SO state
        self.transitions['down_full'] = diff.where(diff==-1,0) #full record of transitions from SO to quiet state
        self.transitions['up_vs_time'] = self.transitions['up_full'].sum(axis=1) #count of transitions from quiet to SO state vs time
        self.transitions['down_vs_time'] = self.transitions['down_full'].sum(axis=1) #count of transitions from SO to quiet state vs time
        self.transitions['up_vs_shot'] = self.transitions['up_full'].sum(axis=0) #count of transitions from quiet to SO state vs shot (i.e. column)
        self.transitions['down_vs_shot'] = self.transitions['down_full'].sum(axis=0) #count of transitions from SO to quiet state vs shot (i.e. column)
        self.transitions['shot_vs_time'] = self.transitions['up_full'].where(self.transitions['up_full'] > 0).idxmax().dropna() #records time of transition only if a transition from quiet to SO occurs
        self.transitions['shot_vs_time_down'] = self.transitions['down_full'].where(self.transitions['down_full'] < 0).idxmin().dropna() #records time of transition only if a transition from SO to quiet occurs

    def plot_IQ_shots(self,save_name=None,**kwargs):

        axes = kwargs.get('axes', None)
        _flag_axes_supplied = True
        if not axes:
            fig, axes = generate_axes(shape=(2,1))
            _flag_axes_supplied = False

        axes[0].imshow(self.IQs,extent=[0,self.nshots,self.time[0],self.time[-1]])
        axes[0].set_xlabel('Shot')
        axes[0].set_ylabel(r'Time ($\mu$s)')

        axes[1].plot(self.IQs,color=et_colors[0],alpha=0.1,lw=0.5)
        axes[1].set_ylabel('|IQ|')
        axes[1].set_xlabel(r'Time ($\mu$s)')

        if _flag_axes_supplied:
            return(axes)
        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        else:
            plt.show()
