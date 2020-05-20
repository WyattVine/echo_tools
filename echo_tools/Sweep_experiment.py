import pandas as pd
import numpy as np
import scipy as sp
from scipy import optimize
import matplotlib.pyplot as plt
import warnings
from .utilities import *
from . Echo_experiment import *
from .Echo_trace import *


class Sweep_experiment(Echo_experiment):

    '''
    A class for experiments where IQ traces are collected as a function of some 1D sweep parameter (e.g. pulse power)
    Provides simple way to trim the data, subtract the baseline on I and Q, integrate the echos, and plot the data
    '''

    def __init__(self,data_loc=None,save_loc=None,**kwargs):

        super().__init__(data_loc,save_loc)
        self.data_name_convention = kwargs.get('data_name_convention', 'Is')
        self.data_file_type = kwargs.get('data_file_type', 'pkl')
        self.sweep_parameter = kwargs.get('sweep_parameter', None)

        if data_loc: #allows user to manually specify Is and Qs instead of using Echo_experiment.read()
            self.read_data()

    def trim(self,t1,t2):
        '''
        trims self.Is and self.Qs to only include times between t1 and t2 (e.g. to cut out ringdown)
        '''

        self.Is = self.Is.loc[t1:t2,:]
        self.Qs = self.Qs.loc[t1:t2, :]
        self.time = np.array(self.Is.index)

    def plot_2D(self,**kwargs):
        '''
        Creates 2D colorplots of I, Q, and IQ
        If given save_name will save pdf of image at self.save_loc
        '''

        axes = kwargs.get('axes', None)  # user can supply fig,[ax1,ax2,ax3] which will be returned to them

        extent = [float(self.columns[0]),float(self.columns[-1]),self.time[0],self.time[-1]]
        IQmags = (self.Is**2 + self.Qs**2).apply(np.sqrt)

        if not axes:
            fig, [ax1,ax2,ax3] = generate_axes(shape=(3,1))
        else:
            fig, [ax1,ax2,ax3] = axes

        for i in zip([ax1,ax2,ax3],[self.Is,self.Qs,IQmags],['I','Q','|IQ|']):
            im = i[0].imshow(i[1],aspect='auto',origin='lower',extent=extent)
            fig.colorbar(im,ax=i[0],shrink=0.8)
            i[0].set_ylabel('Time (us)')
            i[0].set_xlabel(self.sweep_parameter)
            i[0].set_title(i[2])

        if not axes:
            plt.tight_layout()
            save_name = kwargs.get('save_name',None)
            if save_name:
                plt.savefig(self.save_loc + save_name)
                plt.close()
            else:
                plt.show()
        return(fig,[ax1,ax2,ax3])

    def remove_baseline(self,t1,t2,t3,t4,order=1,**kwargs):
        ''''
        Remove baseline from data. Updates self.Is and self.Qs
        t1 - t4: define two regions of baseline on either side of the echo
        order: order of polynomial used in fitting of baseline, defaults to 1 = linear fit
        '''

        Is_corr = pd.DataFrame(index=self.Is.index,columns=self.Is.columns, dtype=np.float64)
        Qs_corr = pd.DataFrame(index=self.Qs.index, columns=self.Qs.columns, dtype=np.float64)
        for col in np.array(self.Is.columns):
            Is_corr.at[:,col] = remove_polynomial_baseline(self.time,np.array(self.Is.loc[:,col]),t1,t2,t3,t4,order)
            Qs_corr.at[:, col] = remove_polynomial_baseline(self.time, np.array(self.Qs.loc[:, col]),t1,t2,t3,t4,order)

        self.Is = Is_corr
        self.Qs = Qs_corr

        if kwargs.get('plot_comparison',False):
            self.plot_traces(plot_raw_data=True,IQ_style='magnitude',**kwargs)

    def integrate_echos(self,noise_range,plot=True,**kwargs):
        ''''
        Integrate I, Q, and IQ signals by creating an Echo_trace for each column
        noise_range = (t1,t2,t3,t4), specifies the region for creating discriminators in each Echo_trace (std deviation of noise)
        '''

        self.integrated_echos = pd.DataFrame(index = self.columns, columns=('I','Q','IQ'), dtype=np.float64)
        self.integrated_echo_uncertainties = pd.DataFrame(index=self.columns, columns=('I','Q','IQ'), dtype=np.float64)

        for i in self.columns:
            S = Echo_trace(self.Is.loc[:, i], self.Qs.loc[:, i])
            S.integrate_echo(noise_range=noise_range)

            # S.plot()

            for col in ['I','Q','IQ']:
                self.integrated_echos.loc[i,col] = S.integrated_echo[col]
                self.integrated_echo_uncertainties.loc[i, col] = S.integrated_echo_uncertainty[col]

        if plot:
            x = [float(i) for i in self.integrated_echos.index]
            fig, [ax1,ax2,ax3] = generate_axes(shape=(3,1))

            for i in zip([ax1, ax2, ax3], ['I', 'Q', 'IQ']):
                i[0].scatter(x,self.integrated_echos.loc[:,i[1]],s=3,color='b')
                _yplus = np.array(self.integrated_echos.loc[:,i[1]]+self.integrated_echo_uncertainties.loc[:,i[1]]/2)
                _yminus = np.array(self.integrated_echos.loc[:,i[1]]-self.integrated_echo_uncertainties.loc[:,i[1]]/2)
                i[0].fill_between(x,_yplus,_yminus,color='b',alpha=0.2)
                i[0].set_ylabel(i[1] + r'  (V$\cdot \mu$s)')
                i[0].set_xlabel(self.sweep_parameter)

            plt.tight_layout()
            save_name = kwargs.get('save_name',None)
            if save_name:
                plt.savefig(self.save_loc + save_name)
                plt.close()
            else:
                plt.show()

    def plot_traces(self,**kwargs):
        '''
         1D plots of I, Q and IQ
         By default the columns are linearly sampled, but specific columns can be plotted by giving a list of column indicies as an argument
        '''

        num_cols = kwargs.get('num_cols',3) #number of columns to plot
        fig, axes = generate_axes(shape=(3,num_cols))

        n = len(self.columns)//(num_cols - 1)
        column_indices = kwargs.get('column_indices',[0] + [n*i for i in range(1,num_cols-1)] + [-1])
        if num_cols != len(column_indices):
            warnings.UserWarning('The number of columns requested does not match the number of column indices given')
            return

        for i in range(num_cols):
            S = Echo_trace(self.Is.iloc[:,column_indices[i]],self.Qs.iloc[:,column_indices[i]],**kwargs)
            _axes = (axes[i],axes[i+num_cols],axes[i+2*num_cols])
            _axes = S.plot(axes=_axes,**kwargs)
            for j in _axes:
                j.set_title(self.sweep_parameter + ' = {}'.format(self.columns[column_indices[i]]))

        if kwargs.get('plot_raw_data',False):
            for i in range(num_cols):
                S = Echo_trace(self.Is_raw.iloc[:, column_indices[i]], self.Qs_raw.iloc[:, column_indices[i]], **kwargs)
                _axes = (axes[i], axes[i + num_cols], axes[i + 2 * num_cols])
                _axes = S.plot(axes=_axes, **kwargs)

        plt.tight_layout()
        save_name = kwargs.get('save_name',None)
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        else:
            plt.show()

    def rename_columns(self,new_columns):
        '''
        Rename the column indicies of self.Is and self.Qs
        '''

        if len(self.columns) != len(new_columns):
            raise ValueError('Number of new column names provided is {}, '
                             'number of column names required is {}'.format(len(new_columns),len(self.columns)))

        _map = {self.columns[i]:new_columns[i] for i in range(len(self.columns))}
        self.Is = self.Is.rename(columns=_map)
        self.Qs = self.Qs.rename(columns=_map)
        self.columns = new_columns
