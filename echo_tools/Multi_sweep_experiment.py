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

class Multi_sweep_experiment():

    '''
    Compares sweep experiments that are collected under similar conditions (e.g. same pulse sequence
    but with different powers)
    '''

    def __init__(self,data_loc=None,save_loc=None,sweep_labels=None,**kwargs):

        self.data_loc = data_loc
        self.save_loc = save_loc

        self.sweep_parameter = kwargs.get('sweep_parameter',None)

        self.sweeps = {i : Sweep_experiment(data_loc=self.data_loc, save_loc=self.save_loc, sweep_parameter=self.sweep_parameter) for i in sweep_labels}
        self.data_name_conventions = {i : None for i in sweep_labels}
        self.noise_range = kwargs.get('noise_range', None)


    @property
    def sweep_labels(self):
        return self.sweeps.keys()

    @property
    def n_sweeps(self):
        return len(self.sweep_labels)

    @property
    def signals(self):
        return ('I','Q','IQ')

    @property
    def noise_range(self):
        return self._noise_range

    @noise_range.setter
    def noise_range(self,new_range):
        self._noise_range = new_range
        for exp in self.sweeps.values():
            exp.noise_range = new_range

    @property
    def integrated_echos(self):
        df = pd.DataFrame(columns=[],index=self.sweeps.values[0].integrated_echos.index)
        for i,j in self.sweeps.items():
            for k in self.signals:
                df.loc[:,'{} : {}'.format(i,k)] = j.integrated_echos[k]
        return df

    @property
    def integrated_echos(self):
        df = pd.DataFrame(columns=[],index=self.sweeps.values[0].integrated_echos.index)
        for i,j in self.sweeps.items():
            for k in self.signals:
                df.loc[:,'{} : {}'.format(i,k)] = j.echo_amplitudes[k]
        return df

    def read_data(self):
        if list(self.data_name_conventions.values())[0]:
            for i in self.sweep_labels:
                self.sweeps[i].data_name_convention = self.data_name_conventions[i]
                self.sweeps[i].read_data()

    def trim(self,t1,t2):
        for exp in self.sweeps.values():
            exp.trim(t1,t2)

    def remove_baseline(self,order=1,**kwargs):
        for exp in self.sweeps.values():
            exp.remove_baseline(order=order,**kwargs)

    def lowpass_filter(self,order=2,cutoff=500e3,**kwargs):
        for exp in self.sweeps.values():
            exp.lowpass_filter(order=order,cutoff=cutoff,**kwargs)

    def overlay_traces(self,save_name=None,num_cols=3,legend_loc=0,**kwargs):

        fig,axes = generate_axes(shape=(3,num_cols))
        for i in self.sweep_labels:
            self.sweeps[i].plot_traces(axes=axes,num_cols=num_cols,label=i)
        axes[legend_loc].legend()

        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        else:
            plt.show()

    def compare_2D_plots(self,save_name=None,**kwargs):

        fig, axes = generate_axes(shape=(3, self.n_sweeps))
        for i,(label,exp) in enumerate(self.sweeps.items()):
            exp.plot_2D(axes=[axes[i],axes[i+self.n_sweeps],axes[i+2*self.n_sweeps]])
        for i in zip(self.sweep_labels,range(0,self.n_sweeps,3)):
            axes[i[1]].set_title(i[0])

        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        else:
            plt.show()

    def integrate_echos(self,with_discriminators=False,std_multiplier=1,**kwargs):

        for i in list(self.sweeps.values()):
            i.integrate_echos(plot=False,with_discriminators=with_discriminators,std_multiplier=std_multiplier)

    def plot_integrated_echos(self,save_name=None,**kwargs):

        axes = kwargs.get('axes', None)
        _flag_axes_supplied = True
        if not axes:
            fig, axes = generate_axes(shape=(3, 1))
            _flag_axes_supplied = False

        for i,j in self.sweeps.items():
            j.plot_integrated_echos(axes=axes,label=i)

        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        else:
            plt.show()
