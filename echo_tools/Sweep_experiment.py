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
update_matplot_style()
et_colors = color_palette()


class Sweep_experiment(Echo_experiment):

    '''
    A class for experiments where IQ traces are collected as a function of some 1D sweep parameter (e.g. pulse power)
    Provides simple way to trim the data, subtract the baseline on I and Q, integrate the echos, and plot the data
    '''

    def __init__(self,**kwargs):

        super().__init__(**kwargs)
        self.sweep_parameter = kwargs.get('sweep_parameter', None)


    def remove_baseline(self,order=1,**kwargs):
        ''''
        Remove baseline from data. Updates self.Is and self.Qs
        order: order of polynomial used in fitting of baseline, defaults to 1 = linear fit
        '''

        for col in self.columns:
            S = Echo_trace(self.Is.loc[:,col],self.Qs.loc[:,col],noise_range=self.noise_range)
            S.remove_baseline(order=order,**kwargs)
            self.Is.loc[:,col] = np.array(S.data['I'])
            self.Qs.loc[:,col] = np.array(S.data['Q'])


    def plot_2D(self,**kwargs):
        '''
        Creates 2D colorplots of I, Q, and IQ
        '''

        axes = kwargs.get('axes', None)  # user can supply fig,[ax1,ax2,ax3] which will be returned to them
        save_name = kwargs.get('save_name', None)

        if not axes:
            fig, [ax1,ax2,ax3] = generate_axes(shape=(3,1))
        else:
            fig, [ax1,ax2,ax3] = axes

        IQ = self.generate_IQs()
        extent = [float(self.columns[0]), float(self.columns[-1]), self.time[0], self.time[-1]]
        for i in zip([ax1,ax2,ax3],[self.Is,self.Qs,IQ],['I','Q','|IQ|']):
            im = i[0].imshow(i[1],extent=extent)
            fig.colorbar(im,ax=i[0],shrink=0.8)
            i[0].set_ylabel('Time (us)')
            i[0].set_xlabel(self.sweep_parameter)
            i[0].set_title(i[2])

        if not axes:
            plt.tight_layout()

            if save_name:
                plt.savefig(self.save_loc + save_name)
                plt.close()
            else:
                plt.show()
        return(fig,[ax1,ax2,ax3])


    def integrate_echos(self,plot=True,**kwargs):
        ''''
        Integrate I, Q, and IQ signals by creating an Echo_trace for each column
        '''

        self.integrated_echos = pd.DataFrame(index = self.columns, columns=('I','Q','IQ'), dtype=np.float64)
        self.integrated_echo_uncertainties = pd.DataFrame(index=self.columns, columns=('I','Q','IQ'), dtype=np.float64)

        for i in self.columns:
            S = Echo_trace(self.Is.loc[:, i], self.Qs.loc[:, i],noise_range=self.noise_range)
            S.integrate_echo()

            for col in ['I','Q','IQ','|I|','|Q|']:
                self.integrated_echos.loc[i,col] = S.integrated_echo[col]
                self.integrated_echo_uncertainties.loc[i, col] = S.integrated_echo_uncertainty[col]

        if plot:
            self.plot_integrated_echos(**kwargs)


    def plot_integrated_echos(self,save_name=None,**kwargs):
        '''
        Plots integrated echos and their uncertainties
        '''

        axes = kwargs.get('axes',None)
        _flag_axes_supplied = True
        if not axes:
            fig, axes = generate_axes(shape=(3, 1))
            _flag_axes_supplied = False

        x = [float(i) for i in self.integrated_echos.index]
        for i in zip(axes, ['I', 'Q', 'IQ']):
            i[0].plot(x, self.integrated_echos.loc[:, i[1]], 'o', markersize=3)
            _yplus = np.array(self.integrated_echos.loc[:, i[1]] + self.integrated_echo_uncertainties.loc[:, i[1]] / 2)
            _yminus = np.array(self.integrated_echos.loc[:, i[1]] - self.integrated_echo_uncertainties.loc[:, i[1]] / 2)
            i[0].fill_between(x, _yplus, _yminus, alpha=0.2)
            i[0].set_ylabel(i[1] + r'  (V$\cdot \mu$s)')
            i[0].set_xlabel(self.sweep_parameter)

        if _flag_axes_supplied:
            return(axes)

        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        else:
            plt.show()


    def plot_traces(self,num_cols=3,save_name=None,**kwargs):
        '''
         1D plots of I, Q and IQ
         By default the columns are linearly sampled, but specific columns can be plotted by giving a list of column indicies as an argument

         num_cols: int, number of columns to plot
        '''

        n = len(self.columns) // (num_cols - 1)
        column_indices = kwargs.get('column_indices', [0] + [n * i for i in range(1, num_cols - 1)] + [-1])
        IQ_combined = pd.concat((self.Is,self.Qs))
        ylim = [1.05*IQ_combined.min().min(),1.05*IQ_combined.max().max()]

        fig, axes = generate_axes(shape=(3,num_cols))
        if num_cols != len(column_indices):
            warnings.UserWarning('The number of columns requested does not match the number of column indices given')
            return

        for i in range(num_cols):
            S = Echo_trace(self.Is.iloc[:,column_indices[i]],self.Qs.iloc[:,column_indices[i]],**kwargs)
            _axes = (axes[i],axes[i+num_cols],axes[i+2*num_cols])
            _axes = S.plot(axes=_axes,ylim=ylim,**kwargs)
            for j in _axes:
                j.set_title(self.sweep_parameter + ' = {}'.format(self.columns[column_indices[i]]))

        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        else:
            plt.show()






