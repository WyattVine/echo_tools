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
update_matplot_style()
et_colors = color_palette()


class Sweep_2D_experiment():

    def __init__(self,columns,index,data_loc=None,save_loc=None,I_files=None,**kwargs):

        self.data_loc = data_loc
        self.save_loc = save_loc
        self.sweep_parameters = kwargs.get('sweep_parameters',(None,None)) #sweep_parameter within each I_file, sweep_parameter being changed for each I_file
        self.data_file_type = kwargs.get('data_file_type','pkl')
        self._flag_clean_data = False
        self.I_files = I_files

        self.columns = columns #correspond to columns found within each I_file
        self.index = index #correspond to values being changed for each I_file


        @property
        def columns(self):
            return(self._columns)

        @columns.setter
        def columns(self,new_columns):
            if len(self.columns) != len(new_columns):
                raise ValueError('Number of new column names provided is {}, '
                                 'number of column names required is {}'.format(len(new_columns), len(self.columns)))
            self._columns = new_columns

        @property
        def index(self):
            return(self._index)

        @index.setter
        def index(self,new_index):
            if (len(new_index) != self.index) and (len(new_index) != len(self.I_files)):
                raise ValueError('Number of new index names provided is {}, '
                                 'number of index names required is {}'.format(len(new_index), len(self.index)))
            self._index = new_index

        @property
        def data_cleaning_params(self):
            return({'trim_times' : self.trim_times,
                    'noise_range' : self.noise_range,
                    'baseline_poly_order' : self.baseline_poly_order,
                    '_flag_clean_data' : self._flag_clean_data})

        @data_cleaning_params.setter
        def data_cleaning_params(trim_times,noise_range,baseline_poly_order,flag_clean_data=True):
            self.trim_times = trim_times
            self.noise_range = noise_range
            self.baseline_poly_order = baseline_poly_order
            self._flag_clean_data = flag_clean_data

        @property
        def signals(self):
            return('I','Q','IQ')


    def _initialize_Sweep_exp(self,I_file,**kwargs):
        '''
        Given one I_file will instantiate a Sweep_experiment.
        '''

        self._exp = Sweep_experiment(save_loc = self.save_loc,
                               data_loc = self.data_loc,
                               sweep_parameter = self.sweep_parameters[0],
                               data_name_convention = I_file,
                               data_file_type = self.data_file_type,
                               noise_range = self.noise_range)

        self._exp.columns = self.columns

        if self._flag_clean_data:
            self.clean_data()


    def clean_data(self):
        '''
        trims and removes baseline from Sweep_experiment currently open
        '''

        if self._flag_clean_data:
            self._exp.trim(*self.trim_times)
            self._exp.noise_range = self.noise_range,
            self._exp.remove_baseline(order=self.baseline_poly_order)
        else:
            Warning('_flag_clean_data is set to False so data was not cleaned')


    def plot_file(self,file_number,plot_2D = True, plot_traces = True, plot_integrated_echos = True, **kwargs):
        '''
        Uses the Sweep_experiment class to display the data corresponding to I_files[file_number]
        '''

        self._intialize_Sweep_exp(I_file = self.I_files[file_number])
        if plot_2D:
            self.exp.plot_2D()
        if plot_traces:
            self.exp.plot_traces()
        if plot_integrated_echos:
            self.exp.integrate_echos(plot=True)

    def integrate_echos(self,plot=True,**kwargs):

        self.integrated_echos = {i : pd.DataFrame(index=self.columns,columns=self.index,dtype=np.float64) for i in self.signals}

        for i,file in enumerate(self.I_files):

            self._initialize_Sweep_exp(I_file = file)
            self._exp.integrate_echos()
            for j in self.signals:
                self.integrated_echos[j].iloc[:,i] = self._exp.integrated_echos[j]


    def plot_integrated_echos(self,save_name=None):


        axes = kwargs.get('axes',None)
        _flag_axes_supplied = True
        if not axes:
            fig,axes = generate_axes(shape=(3,1))
            _flag_axes_supplied = False

        for i in zip(axes,self.signals):

            im = i[0].imshow(self.integrated_echos[i[1]])

        plt.show()






