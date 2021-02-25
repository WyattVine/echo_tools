import pandas as pd
import numpy as np
import scipy as sp
from scipy import optimize
import copy
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import warnings
from .utilities import *
from .Echo_trace import *
from .fitting_tools import *
update_matplot_style()
et_colors = color_palette()


class Sweep_experiment():

    '''
    A class for experiments where IQ traces are collected as a function of some 1D sweep parameter (e.g. pulse power)
    Provides simple way to trim the data, subtract the baseline on I and Q, integrate the echos, and plot the data
    '''

    def __init__(self,data_loc=None,save_loc=None,read_data=False,**kwargs):

        self.data_loc = data_loc
        self.save_loc = save_loc
        self.sweep_parameter = kwargs.get('sweep_parameter', None)
        self.data_name_convention = kwargs.get('data_name_convention', 'Is')
        self.data_file_type = kwargs.get('data_file_type', 'pkl')
        self.noise_range = kwargs.get('noise_range',None)
        self._flag_baseline_removed = False
        self._flag_lowpass_filtered = False
        self._flag_integrated_with_discriminators = False
        self.integrated_echos = None

        if read_data:
            self.read_data()

    @property
    def signals(self):
        return('I','Q','IQ')

    @property
    def IQs(self):
        return((self.Is ** 2 + self.Qs ** 2).apply(np.sqrt))

    @property
    def IQs_complex(self):
        return self.Is + 1j*self.Qs

    # @property
    # def mean_trace(self):
    #     return Echo_trace(I=self.Is.mean(axis=1),Q=self.Qs.mean(axis=1),save_loc=self.save_loc)

    @property
    def combined_Is_Qs(self):
        return(pd.concat((self.Is, self.Qs)).fillna(0))

    @property
    def max_signal(self):
        return(self.combined_Is_Qs.max().max())

    @property
    def echo_amplitudes(self):
        amps = pd.DataFrame(index=self.columns, columns=self.signals, dtype=np.float64)
        for i in zip(self.signals,[self.Is,self.Qs,self.IQs]):
            amps.loc[:,i[0]] = i[1].apply(np.abs).max()
        return amps

    @property
    def echo_noise_std(self):
        df = pd.DataFrame(index=self.columns, columns=self.signals, dtype=np.float64)
        for i in self.columns:
            S = Echo_trace(self.Is.loc[:,i],self.Qs[:,i])
            S.noise_range = self.noise_range
            for j in self.signals:
                df.loc[i,j] = S.discriminator(j,1)

    @property
    def min_signal(self):
        return (self.combined_Is_Qs.min().min())

    @property
    def columns(self):
        return(np.array(self.Is.columns))

    @columns.setter
    def columns(self,new_columns):
        if len(self.columns) != len(new_columns):
            raise ValueError('Number of new column names provided is {}, '
                             'number of column names required is {}'.format(len(new_columns), len(self.columns)))
        _map = {self.columns[i]: new_columns[i] for i in range(len(self.columns))}
        self.Is = self.Is.rename(columns=_map)
        self.Qs = self.Qs.rename(columns=_map)

    @property
    def time(self):
        return(np.array(self.Is.index))

    @property
    def extent(self):
        return([float(self.columns[0]), float(self.columns[-1]), float(self.time[0]), float(self.time[-1])])

    def read_data(self,**kwargs):
        if self.data_file_type == 'pkl':
            self.Is = pd.read_pickle(self.data_loc + self.data_name_convention + '.pkl')
            self.Qs = pd.read_pickle(self.data_loc + self.data_name_convention.replace('I','Q') + '.pkl')
        elif self.data_file_type == 'csv':
            self.Is = pd.read_csv(self.data_loc + self.data_name_convention + '.csv',index_col=0)
            self.Qs = pd.read_csv(self.data_loc + self.data_name_convention.replace('I','Q') + '.csv',index_col=0)

    def trim(self, t1, t2):
        '''Trims self.Is and self.Qs to only include times between t1 and t2 (e.g. to cut out ringdown)
        and cuts columns with NaN values (indicating a measurement didn't finish)'''
        self.Is = self.Is.loc[t1:t2, :].dropna(axis=1)
        self.Qs = self.Qs.loc[t1:t2, :].dropna(axis=1)

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
        self._flag_baseline_removed = True

    def rotate_onto_I(self,rough=False):

        self.alignment_angles = pd.Series(index=self.columns,dtype=np.float64)
        for col in self.columns:
            S = Echo_trace(self.Is.loc[:,col],self.Qs.loc[:,col])
            S.noise_range = self.noise_range
            theta = S.rotate_onto_I(rough=rough)
            self.Is.loc[:,col] = np.array(S.data['I'])
            self.Qs.loc[:,col] = np.array(S.data['Q'])
            self.alignment_angles[col] = theta

    def lowpass_filter(self,order=2,cutoff=500e3,**kwargs):
        '''
        Apply a digital lowpass filter to the data
        '''
        for col in self.columns:
            S = Echo_trace(self.Is.loc[:,col],self.Qs.loc[:,col],noise_range=self.noise_range)
            S.lowpass_filter(order=order,cutoff=cutoff,**kwargs)
            self.Is.loc[:,col] = S.data['I'].to_numpy()
            self.Qs.loc[:,col] = S.data['Q'].to_numpy()
        self._flag_lowpass_filtered = True

    def fourier_transform(self):

        S = Echo_trace(self.Is.iloc[:,0],self.Qs.iloc[:,0])
        S.fourier_transform(plot=False)
        self.Is_fourier = pd.DataFrame(index=S.fourier_data['freq'],columns=self.columns,dtype=np.float64)
        self.Qs_fourier = pd.DataFrame(index=S.fourier_data['freq'],columns=self.columns,dtype=np.float64)
        for col in self.columns:
            S = Echo_trace(self.Is.loc[:,col],self.Qs.loc[:,col])
            S.fourier_transform(plot=False)
            self.Is_fourier.loc[:,col] = S.fourier_data['I'].to_numpy()
            self.Qs_fourier.loc[:,col] = S.fourier_data['Q'].to_numpy()

    def plot_2D(self,save_name=None,**kwargs):
        '''
        Creates 2D colorplots of I, Q, and IQ
        '''

        axes = kwargs.get('axes', None)
        _flag_axes_supplied = True
        if not axes:
            fig, axes = generate_axes(shape=(3,1))
            _flag_axes_supplied = False

        for i in zip(axes,[self.Is,self.Qs,self.IQs],self.signals):
            im = i[0].imshow(i[1],extent=self.extent)
            if not _flag_axes_supplied:
                clb = fig.colorbar(im,ax=i[0],shrink=0.8)
                clb.ax.set_title('V')
            i[0].set_ylabel(r'Time ($\mu$s)')
            i[0].set_xlabel(self.sweep_parameter)
            i[0].set_title(i[2])

        if _flag_axes_supplied:
            return(axes)
        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        else:
            plt.show()

    def integrate_echos(self,plot=True,with_discriminators=False,std_multiplier=1,**kwargs):
        ''''
        Integrate I, Q, and IQ signals by creating an Echo_trace for each column
        with_discriminators: boolean, integrate the Echo_trace using the method utilizing discriminators or simple sum
        std_multiplier: multiple of std_deviation used for constructing discriminators
        '''

        if not self._flag_baseline_removed:
            Warning('The baseline was not removed prior to integrating')

        self.integrated_echos = pd.DataFrame(index=self.columns, columns=self.signals, dtype=np.float64)

        if with_discriminators:
            self._flag_integrated_with_discriminators = True
            self.integrated_echo_uncertainties = pd.DataFrame(index=self.columns, columns=self.signals, dtype=np.float64)
            for i in self.columns:
                S = Echo_trace(self.Is.loc[:, i], self.Qs.loc[:, i],noise_range=self.noise_range)
                S.integrate_echo_with_discriminators(std_multiplier=std_multiplier,**kwargs)
                for col in [*self.signals,'|I|','|Q|']:
                    self.integrated_echos.loc[i,col] = S.integrated_echo[col]
                    self.integrated_echo_uncertainties.loc[i, col] = S.integrated_echo_uncertainty[col]

        else:
            self._flag_integrated_with_discriminators = False
            for i in self.columns:
                S = Echo_trace(self.Is.loc[:, i], self.Qs.loc[:, i], noise_range=self.noise_range)
                S.integrate_echo(**kwargs)
                for col in [*self.signals, '|I|', '|Q|']:
                    self.integrated_echos.loc[i, col] = S.integrated_echo[col]

        if plot:
            self.plot_integrated_echos(**kwargs)

    def plot_integrated_echos(self,save_name=None,**kwargs):
        '''
        Plots integrated echos and their uncertainties
        '''

        label = kwargs.get('label',None)
        axes = kwargs.get('axes',None)
        _flag_axes_supplied = True
        if not axes:
            fig, axes = generate_axes(shape=(3, 1))
            _flag_axes_supplied = False

        x = [float(i) for i in self.integrated_echos.index]
        for i in zip(axes, self.signals):
            i[0].plot(x, self.integrated_echos.loc[:, i[1]], 'o', label=label)
            if self._flag_integrated_with_discriminators:
                _yplus = np.array(self.integrated_echos.loc[:, i[1]] + self.integrated_echo_uncertainties.loc[:, i[1]])
                _yminus = np.array(self.integrated_echos.loc[:, i[1]] - self.integrated_echo_uncertainties.loc[:, i[1]])
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

    def plot_echo_amplitudes(self,save_name=None,**kwargs):
        '''
        Plots maximum echo amplitudes and their uncertainties
        '''

        label = kwargs.get('label',None)
        axes = kwargs.get('axes',None)
        _flag_axes_supplied = True
        if not axes:
            fig, axes = generate_axes(shape=(3, 1))
            _flag_axes_supplied = False

        amps = self.echo_amplitudes
        uncertainties = self.echo_noise_std
        x = [float(i) for i in amps.index]
        for i in zip(axes, self.signals):
            i[0].plot(x, amps.loc[:, i[1]], 'o', label=label)

            _yplus = np.array(self.echo_amplitudes.loc[:, i[1]] + uncertainties.loc[:, i[1]])
            _yminus = np.array(self.echo_amplitudes.loc[:, i[1]] - uncertainties.loc[:, i[1]])
            i[0].fill_between(x, _yplus, _yminus, alpha=0.2)

            i[0].set_ylabel(i[1] + r'(V)')
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
        if num_cols != len(column_indices):
            warnings.UserWarning('The number of columns requested does not match the number of column indices given')
            return

        ylim = [1.05*self.min_signal,1.05*self.max_signal]
        label = kwargs.get('label',None)
        axes = kwargs.get('axes',None)
        _flag_axes_supplied = True
        if not axes:
            fig,axes = generate_axes(shape=(3,num_cols))
            _flag_axes_supplied = False

        for i in range(num_cols):
            S = Echo_trace(self.Is.iloc[:,column_indices[i]],self.Qs.iloc[:,column_indices[i]],**kwargs)
            _axes = (axes[i],axes[i+num_cols],axes[i+2*num_cols])
            _axes = S.plot(axes=_axes,ylim=ylim,label=label)
            for j in _axes:
                j.set_title(self.sweep_parameter + ' = {}'.format(self.columns[column_indices[i]]))

        if _flag_axes_supplied:
            return(axes)
        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        else:
            plt.show()

    def plot_temperatures(self,file,**kwargs):

        self.temps = pd.read_pickle(file)

        fig,axes = generate_axes(shape=(1,3),figsize=(9,3))
        for i in zip(axes,['He3_pot','He4_pot','He3_sorb']):
            i[0].plot(self.temps[i[1]])
            i[0].set_xlabel('Measurment Number')
            i[0].set_ylabel('Temperature (K)')
            i[0].set_title(i[1])
        plt.tight_layout()
        plt.savefig(self.save_loc + 'temperatures.png',dpi=300,bbox_inches='tight')
        plt.close()

    def make_trace(self,col_idx):

        if col_idx == 'mean':
            return Echo_trace(self.Is.mean(axis=1),self.Qs.mean(axis=1),save_loc=self.save_loc)
        else:
            return Echo_trace(self.Is.iloc[:,col_idx],self.Qs.iloc[:,col_idx],save_loc=self.save_loc)



class Average_sweep_experiment():

    def __init__(self,reps):
        self.reps = reps
        self.nreps = len(reps)

        self.average_reps()


    def average_reps(self):
        self.avg = copy.deepcopy(self.reps[0])
        for i in range(1,self.nreps):
            self.avg.Is += self.reps[i].Is.copy()
            self.avg.Qs += self.reps[i].Qs.copy()
        self.avg.Is /= self.nreps
        self.avg.Qs /= self.nreps
        if not self.avg.integrated_echos is None:
            self._unify_integrated_echos()


    def _unify_integrated_echos(self):
        for i,j in enumerate(self.reps):
            j.integrated_echos['rep'] = [i for k in range(len(j.columns))]

        self.integrated_echos = pd.concat([i.integrated_echos for i in self.reps])
        self.integrated_echos[self.avg.sweep_parameter] = self.integrated_echos.index
