import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .utilities import *

class Echo_trace():
    
    '''
    Basic representation of a single echo time trace.
    '''
    
    def __init__(self,I,Q,time=None,**kwargs):
        
        '''
        time = np.ndarray or pd.core.series.Series
        I = np.ndarray or pd.core.series.Series
        Q = np.ndarray or pd.core.series.Series
        '''


        self.data = pd.DataFrame(columns=('time', 'I', 'Q', 'S', 'IQ'))
        if (type(I) == type(Q)) and (type(I) == np.ndarray):
            self.data['time'] = time
            self.data['I'] = I
            self.data['Q'] = Q
        elif (type(I) == type(Q)) and (type(I) == pd.core.series.Series):
            self.data['time'] = np.array(I.index)
            self.data['I'] = np.array(I)
            self.data['Q'] = np.array(Q)

        self.data['S'] = self.data['I'] + 1j*self.data['Q']
        self.data['IQ'] = np.abs(self.data['S'])
        self.dt = self.data['time'].iloc[2] - self.data['time'].iloc[1]
        self._flag_discriminators = False #True when discriminators have been created
        self.save_loc = kwargs.get('save_loc',None)
        
    def rotate(self,theta):
        '''
        Rotates the echo trace by theta (radians) in complex plane 
        '''
        
        self.data['S'] = self.data['S']*np.exp(1j*theta)
        self.data['I'] = np.array(self.data['S']).real
        self.data['Q'] = np.array(self.data['S']).imag
        
    def plot(self,**kwargs):

        min_max_times = [self.data['time'].min(),self.data['time'].max()]
        I_lims = kwargs.get('I_lims',[-0.5,0.5])
        Q_lims = kwargs.get('Q_lims', [-0.5, 0.5])
        IQ_lims = kwargs.get('IQ_lims',[-0.05,1.0])
        IQ_style = kwargs.get('IQ_style','complex_circle') #'complex_circle' or 'magnitude'
        axes = kwargs.get('axes',None) #user can supply ax1,ax2,ax3 which will be returned to them
        label = kwargs.get('label',None)

        if not axes:
            fig, [ax1,ax2,ax3] = generate_axes(shape=(3,1))
        else:
            ax1,ax2,ax3 = axes

        for i in zip([ax1,ax2],['I','Q'],[I_lims,Q_lims],['I (V)','Q (V)']):
            i[0].plot(self.data['time'],self.data[i[1]],label=label)
            i[0].set_xlabel('Time (us)')
            i[0].plot(min_max_times,[0,0],c='black',alpha=0.3)
            i[0].set_xlim(min_max_times)
            i[0].set_ylim(i[2])
            i[0].set_ylabel(i[3])

        if IQ_style == 'complex_circle':
            ax3.plot(self.data['I'],self.data['Q'],label=label)
            ax3.set_xlim(I_lims)
            ax3.set_ylim(Q_lims)
            ax3.plot(I_lims,[0,0],c='black',alpha=0.3)
            ax3.plot([0,0],Q_lims,c='black',alpha=0.3)
            ax3.set_xlabel('I (V)')
            ax3.set_ylabel('Q (V)')
        elif IQ_style == 'magnitude':
            ax3.plot(self.data['time'],self.data['IQ'],label=label)
            ax3.plot(min_max_times, [0,0], c='black', alpha=0.3)
            ax3.set_xlim(min_max_times)
            ax3.set_ylim(IQ_lims)
            ax3.set_xlabel('Time (us)')
            ax3.set_ylabel('|IQ| (V)')

        if self._flag_discriminators:
            ax1.fill_between(min_max_times,[self.discriminators['I'],self.discriminators['I']],[-1*self.discriminators['I'],-1*self.discriminators['I']],color='r',alpha=0.2)
            ax2.fill_between(min_max_times,[self.discriminators['Q'],self.discriminators['Q']],[-1*self.discriminators['Q'],-1*self.discriminators['Q']],color='r',alpha=0.2)
            if IQ_style == 'complex_circle':
                discriminator_circle = circle(r=self.discriminators['IQ'])
                ax3.plot(discriminator_circle.coords.real, discriminator_circle.coords.imag, c='r')
            elif IQ_style == 'magnitude':
                ax3.fill_between(min_max_times, [self.discriminators['IQ'], self.discriminators['IQ']],[0,0], color='r', alpha=0.2)

        if not axes:
            plt.tight_layout()
            save_name = kwargs.get('save_name',None)
            if save_name:
                plt.savefig(self.save_loc + save_name)
            else:
                plt.show()
        else:
            return(ax1,ax2,ax3)
        
    def create_discriminators(self,t1,t2,t3=None,t4=None,**kwargs):
        '''
        Using t1 - t4 it creates single values corresponding to the noise in I, Q, and IQ that can be used for
        slicing self.data.

        std_mutliplier : discriminators are multiples of the standard deviation in each signal. Supplied via kwargs
        '''

        std_multiplier = kwargs.get('std_multiplier',1)

        _generate_reduced = lambda ta,tb:  self.data[(self.data['time'] >= ta) & (self.data['time'] <= tb)]
        _reduced = _generate_reduced(t1,t2)
        if t3:
            _reduced = pd.concat((_reduced,_generate_reduced(t3,t4)))
            
        self.discriminators = {}
        for i in ['I','Q','IQ']:
            self.discriminators[i] = std_multiplier*np.std(_reduced[i])

        self._flag_discriminators = True
        
    def integrate_echo(self,**kwargs):

        noise_range = kwargs.get('noise_range',None) #tuple of t1,t2,t3,t4
        if noise_range:
            self.create_discriminators(*noise_range)

        _IQ = self.data[self.data['IQ'] > self.discriminators['IQ']]['IQ']
        _I = self.data[(self.data['I'] > self.discriminators['I']) | (self.data['I'] < -1*self.discriminators['I'])]['I']
        _Q = self.data[(self.data['Q'] > self.discriminators['Q']) | (self.data['Q'] < -1*self.discriminators['Q'])]['Q']
        
        self.integrated_echo = {}
        self.integrated_echo['IQ'] = (_IQ - self.discriminators['IQ']).sum()*self.dt
        self.integrated_echo['I'] = np.abs(_I).sum()*self.dt
        self.integrated_echo['Q'] = np.abs(_Q).sum()*self.dt

        self.integrated_echo_uncertainty = {}
        for i in zip([_I,_Q,_IQ],['I','Q','IQ']):
            self.integrated_echo_uncertainty[i[1]] = i[0].count()*self.discriminators[i[1]]*self.dt