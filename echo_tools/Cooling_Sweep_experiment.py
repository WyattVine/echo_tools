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

class Cooling_Sweep_experiment():

    '''
    For experiments where a cooling pump is applied and compared to a single/averaged measurement without the pump

    Instantiates a Sweep_experiment for the data with pump on and an Echo_trace for the data with pump off
    '''

    def __init__(self,data_loc=None,save_loc=None,read_data=False,**kwargs):

        self.pump_on = Sweep_experiment()
        self.pump_off = Echo_trace()

        self.data_loc = data_loc
        self.save_loc = save_loc
        self.noise_range = kwargs.get('noise_range', None)
        self.data_name_convention_pump_on = kwargs.get('data_name_convention_pump_on', 'Is_on')
        self.data_name_convention_pump_off = kwargs.get('data_name_convention_pump_off','I_off')
        self.sweep_parameter = kwargs.get('sweep_parameter',None)

        if read_data:
            self.read_data()


    @property
    def data_loc(self):
        return(self._data_loc)

    @data_loc.setter
    def data_loc(self,loc):
        self._data_loc = loc
        self.pump_on.data_loc = loc
        self.pump_off.data_loc = loc

    @property
    def save_loc(self):
        return (self._save_loc)

    @save_loc.setter
    def save_loc(self, loc):
        self._save_loc = loc
        self.pump_on.save_loc = loc
        self.pump_off.save_loc = loc

    @property
    def columns(self):
        return(self.pump_on.columns)

    @columns.setter
    def columns(self,cols):
        self.pump_on.columns = cols

    @property
    def noise_range(self):
        return(self._noise_range)

    @noise_range.setter
    def noise_range(self,range):
        self._noise_range = range
        self.pump_on.noise_range = range
        self.pump_off.noise_range = range

    @property
    def data_name_convention_pump_on(self):
        return(self.pump_on.data_name_convention)

    @data_name_convention_pump_on.setter
    def data_name_convention_pump_on(self,convention):
        self.pump_on.data_name_convention = convention

    @property
    def data_name_convention_pump_off(self):
        return(self.pump_off.data_name_convention)

    @data_name_convention_pump_off.setter
    def data_name_convention_pump_off(self, convention):
        self.pump_off.data_name_convention = convention

    @property
    def sweep_parameter(self):
        return(self.pump_on.sweep_parameter)

    @sweep_parameter.setter
    def sweep_parameter(self,parameter):
        self.pump_on.sweep_parameter = parameter

    @property
    def max_signal(self):
        return(max(self.pump_off.max_signal,self.pump_on.max_signal))

    @property
    def min_signal(self):
        return(min(self.pump_off.min_signal,self.pump_on.min_signal))

    @property
    def signals(self):
        return('I','Q','IQ')

    @property
    def integrated_echos(self):
        return({'pump_off':self.pump_off.integrated_echo, 'pump_on':self.pump_on.integrated_echos})

    @property
    def integrated_echo_uncertainties(self):
        return ({'pump_off': self.pump_off.integrated_echo_uncertainty, 'pump_on': self.pump_on.integrated_echo_uncertainties})


    def read_data(self):
        self.pump_on.read_data()
        self.pump_off.read_data()


    def trim(self,t1,t2):
        self.pump_on.trim(t1,t2)
        self.pump_off.trim(t1,t2)


    def remove_baseline(self,order=1,**kwargs):
        self.pump_on.remove_baseline(order=order,**kwargs)
        self.pump_off.remove_baseline(order=order,**kwargs)


    def plot_traces(self,num_cols=3,**kwargs):
        ''''
        Overlays traces of pump_on and pump_off
        '''

        fig,axes = generate_axes(shape=(3,num_cols))
        axes = self.pump_on.plot_traces(num_cols=num_cols,axes=axes,label='Pump On')
        for i in range(num_cols):
            _axes = (axes[i],axes[i+num_cols],axes[i+2*num_cols])
            self.pump_off.plot(axes=_axes,label='Pump Off')

        axes[0].legend()
        for ax in axes:
            ax.set_ylim([1.05*self.min_signal,1.05*self.max_signal])

        plt.tight_layout()
        plt.show()


    def integrate_echos(self,std_multiplier=1,plot=True,**kwargs):
        self.pump_on.integrate_echos(std_multiplier=std_multiplier,plot=False)
        self.pump_off.integrate_echo(std_multiplier=std_multiplier)
        if plot:
            self.plot_integrated_echos(**kwargs)


    def plot_integrated_echos(self,save_name=None,**kwargs):
        '''
        Plots a comparison of the integrated echos for pump_on and pump_off
        '''

        axes = kwargs.get('axes', None)
        _flag_axes_supplied = True
        if not axes:
            fig, axes = generate_axes(shape=(3, 1))
            _flag_axes_supplied = False

        axes = self.pump_on.plot_integrated_echos(axes=axes,label='Pump On')
        for i in zip(axes,self.signals):
            x = (self.pump_on.columns[0],self.pump_on.columns[-1])
            y = np.array([self.pump_off.integrated_echo[i[1]] for j in range(2)])
            i[0].plot(x,y,label='Pump Off')
            i[0].fill_between(x,y+self.pump_off.integrated_echo_uncertainty[i[1]],y-self.pump_off.integrated_echo_uncertainty[i[1]],alpha=0.2)
        axes[0].legend()

        if _flag_axes_supplied:
            return(axes)
        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        plt.show()


    def plot_2D_difference(self,save_name=None,**kwargs):
        '''
        Plots the difference between pump on and pump off signals using imshow.
        '''

        axes = kwargs.get('axes', None)
        _flag_axes_supplied = True
        if not axes:
            fig, axes = generate_axes(shape=(3, 1))
            _flag_axes_supplied = False

        for i in zip(axes,(self.pump_on.Is,self.pump_on.Qs,self.pump_on.IQs),self.signals):

            im = i[0].imshow(i[1].subtract(np.array(self.pump_off.data[i[2]]),axis='index'),extent=self.pump_on.extent)
            clb = fig.colorbar(im, ax=i[0], shrink=0.8)
            clb.ax.set_title('V')
            i[0].set_xlabel(self.sweep_parameter)
            i[0].set_ylabel(r'Time ($\mu$s)')
            i[0].set_title(i[2])

        if _flag_axes_supplied:
            return(axes)
        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        plt.show()




