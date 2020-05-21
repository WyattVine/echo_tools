import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .utilities import *
update_matplot_style()

et_colors = color_palette()

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

        self.save_loc = kwargs.get('save_loc', None)

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
        self.discriminators = {} #record values used to discriminate noise from signal (used in integration)
        self._flag_discriminators = False #True when discriminators have been created
        self.noise_range = kwargs.get('noise_range',None) #(t1,t2,t3,t4) define two continuous segments of baseline noise

    def rotate(self,theta):
        '''
        Rotates the echo trace by theta (radians) in complex plane
        '''

        self.data['S'] = self.data['S']*np.exp(1j*theta)
        self.data['I'] = np.array(self.data['S']).real
        self.data['Q'] = np.array(self.data['S']).imag

    def trim(self,t1,t2):
        '''
        Trims self.Is and self.Qs to only include times between t1 and t2 (e.g. to cut out ringdown)
        '''

        self.data = self.data[(self.data['time'] > t1) & (self.data['time'] < t2)]

    def remove_baseline(self,order=1):
        '''
        Removes the baseline of the data using noise_range to specify the values that should be centered at 0
        '''

        for col in ['I','Q']:
            self.data.loc[:,col] = remove_polynomial_baseline(self.data['time'],np.array(self.data[col]),*self.noise_range,order)
        self.data['S'] = self.data['I'] + 1j*self.data['Q']
        self.data['IQ'] = np.abs(self.data['S'])

    def plot(self,**kwargs):

        min_max_times = [self.data['time'].min(),self.data['time'].max()]
        I_lims = kwargs.get('I_lims',[-1,1])
        Q_lims = kwargs.get('Q_lims', [-1, 1])
        IQ_lims = kwargs.get('IQ_lims',[-0.05,1.0])
        IQ_style = kwargs.get('IQ_style','magnitude') #'complex_circle' or 'magnitude'
        axes = kwargs.get('axes',None) #user can supply ax1,ax2,ax3 which will be returned to them
        label = kwargs.get('label',None)
        save_name = kwargs.get('save_name', None)

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

        _generate_reduced = lambda ta,tb:  self.data[(self.data['time'] >= ta) & (self.data['time'] <= tb)]
        _reduced = pd.concat((_generate_reduced(*self.noise_range[:2]),_generate_reduced(*self.noise_range[2:])))


        for i in ['I','Q','IQ']:
            self.discriminators[i] = std_multiplier*np.std(_reduced[i])

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

def compare_traces(trace1,trace2,**kwargs):
    '''
    Overlays Echo_trace.plot of the two Echo_trace objects for a direct comparison
    '''

    I_lims = kwargs.get('I_lims',[-1,1])
    Q_lims = kwargs.get('Q_lims',[-1,1])
    IQ_style = kwargs.get('IQ_style','magnitude')
    labels = kwargs.get('labels',(None,None))
    legend = kwargs.get('legend',False)
    save_name = kwargs.get('save_name',None)
    axes = kwargs.get('axes',None)


    _flag_axis_supplied = True
    if not axes:
        fig,axes = generate_axes(shape=(3,1))
        _flag_axis_supplied = False

    trace1.plot(axes=axes,I_lims=I_lims,Q_lims=Q_lims,IQ_style=IQ_style,label=labels[0])
    trace2.plot(axes=axes,I_lims=I_lims,Q_lims=Q_lims,IQ_style=IQ_style,label=labels[1])

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
