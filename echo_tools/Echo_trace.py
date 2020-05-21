import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .utilities import *
from .fitting_tools import *
update_matplot_style()
et_colors = color_palette()

class Echo_trace():

    '''
    Basic representation of a single echo time trace. Used
    '''

    def __init__(self,I,Q,time=None,**kwargs):
        '''
        time = np.ndarray or pd.core.series.Series
        I = np.ndarray or pd.core.series.Series
        Q = np.ndarray or pd.core.series.Series
        '''

        self.save_loc = kwargs.get('save_loc', None)

        self.data = pd.DataFrame(columns=('time', 'I', 'Q', 'IQ'),dtype=np.float64)
        if (type(I) == type(Q)) and (type(I) == np.ndarray):
            self.data['time'] = time
            self.data['I'] = I
            self.data['Q'] = Q
        elif (type(I) == type(Q)) and (type(I) == pd.core.series.Series):
            self.data['time'] = np.array(I.index)
            self.data['I'] = np.array(I)
            self.data['Q'] = np.array(Q)
        self.data['IQ'] = np.sqrt(self.data['I']**2 + self.data['Q']**2)

        self.dt = self.data['time'].iloc[2] - self.data['time'].iloc[1]
        self.discriminators = {} #record values used to discriminate noise from signal (used in integration)
        self._flag_discriminators = False #True when discriminators have been created
        self.noise_range = kwargs.get('noise_range',None) #(t1,t2,t3,t4) define two continuous segments of baseline noise


    def rotate(self,theta):
        '''
        Rotates the echo trace by theta (radians) in complex plane
        '''

        S = np.array(self.data['I'] + 1j*self.data['Q'])*np.exp(1j*theta)
        self.data['I'] = S.real
        self.data['Q'] = S.imag


    def trim(self,t1,t2):
        '''
        Trims self.Is and self.Qs to only include times between t1 and t2 (e.g. to cut out ringdown)
        '''

        self.data = self.data[(self.data['time'] > t1) & (self.data['time'] < t2)]


    def remove_baseline_old(self,order=1):
        '''
        Removes the baseline of the data using noise_range to specify the values that should be centered at 0
        '''

        for col in ['I','Q']:
            self.data.loc[:,col] = remove_polynomial_baseline(self.data['time'],np.array(self.data[col]),*self.noise_range,order)
        self.data['IQ'] = np.sqrt(self.data['I']**2 + self.data['Q']**2)


    def remove_baseline(self,order=1,**kwargs):
        '''
        Removes the baseline of the data using noise_range to specify the values that should be centered at 0

        order: order of polynomial used to fit baseline. (1, 2 or 3)
        '''

        fit_classes = {1: Linear_fit, 2: Quadratic_fit, 3: Cubic_fit}
        noise_data = pd.concat((self.select_time_range(*self.noise_range[:2]),self.select_time_range(*self.noise_range[2:])))
        self.baseline_fits = {i : fit_classes[order](noise_data['time'],noise_data[i],**kwargs) for i in ['I','Q']}

        for key,val in self.baseline_fits.items():
            self.data[key] = self.data[key] - np.array([val.function(i,*val.params) for i in self.data['time']])
        self.data['IQ'] = np.sqrt(self.data['I']**2 + self.data['Q']**2)


    def plot(self,save_name=None,**kwargs):
        '''
        Plots I, Q and IQ
        Can evaulate descriminators/integration by plotting after creating discriminators (_flag_discriminators = True)
        '''

        xlim = kwargs.get('xlim',[self.data['time'].min(),self.data['time'].max()])
        ylim = kwargs.get('ylim',[1.05*self.data[['I','Q']].min().min(),1.05*self.data[['I','Q']].max().max()])
        IQ_style = kwargs.get('IQ_style','magnitude') #'complex_circle' or 'magnitude'
        axes = kwargs.get('axes',None) #user can supply ax1,ax2,ax3 which will be returned to them
        label = kwargs.get('label',None)

        if not axes:
            fig, [ax1,ax2,ax3] = generate_axes(shape=(3,1))
        else:
            ax1,ax2,ax3 = axes

        for i in zip([ax1,ax2],['I','Q'],['I (V)','Q (V)']):
            i[0].plot(self.data['time'],self.data[i[1]],label=label)
            i[0].set_xlabel('Time (us)')
            i[0].plot(xlim,[0,0],c='black',alpha=0.3)
            i[0].set_xlim(xlim)
            i[0].set_ylim(ylim)
            i[0].set_ylabel(i[2])
        if IQ_style == 'complex_circle':
            ax3.plot(self.data['I'],self.data['Q'],label=label)
            ax3.set_xlim(ylim)
            ax3.set_ylim(ylim)
            ax3.plot(xlim,[0,0],c='black',alpha=0.3)
            ax3.plot([0,0],ylim,c='black',alpha=0.3)
            ax3.set_xlabel('I (V)')
            ax3.set_ylabel('Q (V)')
        elif IQ_style == 'magnitude':
            ax3.plot(self.data['time'],self.data['IQ'],label=label)
            ax3.plot(xlim, [0,0], c='black', alpha=0.3)
            ax3.set_xlim(xlim)
            ax3.set_ylim([-0.05,1.05*self.data['IQ'].max()])
            ax3.set_xlabel('Time (us)')
            ax3.set_ylabel('|IQ| (V)')

        if self._flag_discriminators:
            ax1.fill_between(xlim,[self.discriminators['I'],self.discriminators['I']],[-1*self.discriminators['I'],-1*self.discriminators['I']],color='r',alpha=0.2)
            ax2.fill_between(xlim,[self.discriminators['Q'],self.discriminators['Q']],[-1*self.discriminators['Q'],-1*self.discriminators['Q']],color='r',alpha=0.2)
            if IQ_style == 'complex_circle':
                discriminator_circle = circle(r=self.discriminators['IQ'])
                ax3.plot(discriminator_circle.coords.real, discriminator_circle.coords.imag, c='r')
            elif IQ_style == 'magnitude':
                ax3.fill_between(xlim, [self.discriminators['IQ'], self.discriminators['IQ']],[0,0], color='r', alpha=0.2)

        if not axes:
            plt.tight_layout()
            if save_name:
                plt.savefig(self.save_loc + save_name)
                plt.close()
            else:
                plt.show()
        else:
            return(ax1,ax2,ax3)


    def create_discriminators(self,**kwargs):
        '''
        Using self.noise_range create single values corresponding to the noise in I, Q, and IQ
        (e.g. for use in slicing self.data)

        std_mutliplier : discriminators are multiples of the standard deviation in each signal. Supplied via kwargs
        '''

        std_multiplier = kwargs.get('std_multiplier',1)
        noise = pd.concat((self.select_time_range(*self.noise_range[:2]),self.select_time_range(*self.noise_range[2:])))
        for i in ['I','Q','IQ']:
            self.discriminators[i] = std_multiplier*np.std(noise[i])
        self._flag_discriminators = True


    def integrate_echo(self,**kwargs):
        '''
        Integrates the echo signals by filtering summing the data in self.data that is filtered based on the
        discriminators
        '''

        if self.noise_range and not self._flag_discriminators != 0:
            self.create_discriminators()
        else:
            print('Must specify noise_range before integrating.')
            return

        _IQ = self.data[self.data['IQ'] > self.discriminators['IQ']]['IQ']
        _I = self.data[(self.data['I'] > self.discriminators['I']) | (self.data['I'] < -1*self.discriminators['I'])]['I']
        _Q = self.data[(self.data['Q'] > self.discriminators['Q']) | (self.data['Q'] < -1*self.discriminators['Q'])]['Q']

        self.integrated_echo = {}
        self.integrated_echo['I'] = _I.sum() * self.dt
        self.integrated_echo['Q'] = _Q.sum() * self.dt
        self.integrated_echo['IQ'] = (_IQ - self.discriminators['IQ']).sum()*self.dt
        self.integrated_echo['|I|'] = np.abs(_I).sum()*self.dt
        self.integrated_echo['|Q|'] = np.abs(_Q).sum()*self.dt

        self.integrated_echo_uncertainty = {}
        for i in zip([_I,_Q,_IQ],['I','Q','IQ']):
            self.integrated_echo_uncertainty[i[1]] = i[0].count()*self.discriminators[i[1]]*self.dt
        for i in zip(['I','Q'],['|I|','|Q|']):
            self.integrated_echo_uncertainty[i[1]] = self.integrated_echo_uncertainty[i[0]]


    def select_time_range(self,ta,tb):
        '''
        Returns a copy of self.data between times ta and tb
        '''

        selection = self.data[(self.data['time'] >= ta) & (self.data['time'] <= tb)].copy()
        return(selection)


    def compare_with_trace(self,trace,save_name=None,**kwargs):
        '''
        Overlays Echo_trace.plot of the two Echo_trace objects for a direct comparison
        '''

        labels = kwargs.get('labels',(None,None))
        legend = kwargs.get('legend',False)
        axes = kwargs.get('axes',None)

        _flag_axis_supplied = True
        if not axes:
            fig,axes = generate_axes(shape=(3,1))
            _flag_axis_supplied = False

        self.plot(axes=axes,label=labels[0])
        trace.plot(axes=axes,label=labels[1])

        if legend:
            for i in axes:
                i.legend()

        if _flag_axis_supplied:
            return(axes)

        plt.tight_layout()
        if save_name:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()
