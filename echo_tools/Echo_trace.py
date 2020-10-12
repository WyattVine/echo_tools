import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
from scipy import fftpack
import sklearn.decomposition
from .utilities import *
from .fitting_tools import *
update_matplot_style()
et_colors = color_palette()

class Echo_trace():

    '''
    Basic representation of a single echo time trace. Used
    '''

    def __init__(self,I=pd.Series(dtype=np.float64),Q=pd.Series(dtype=np.float64),data_loc=None,save_loc=None,read_data=False,**kwargs):
        '''
        I = pd.core.series.Series
        Q = pd.core.series.Series
        I and Q can be supplied as arguments or read in by specifying data_loc and file name convention
        '''

        self.save_loc = save_loc
        self.data_loc = data_loc
        self.data_name_convention = kwargs.get('data_name_convention','I')
        self.data_file_type = kwargs.get('data_file_type','pkl')
        self.noise_range = kwargs.get('noise_range',None)  # (t1,t2,t3,t4) define two continuous segments of baseline noise

        if read_data:
            self.read_data()
        elif (not I.empty) and (not Q.empty):
            self._create_DataFrame(I,Q)


    @property
    def signals(self):
        return('I','Q','IQ')

    @property
    def IQ(self):
        return(np.sqrt(self.data['I'] ** 2 + self.data['Q'] ** 2))

    @property
    def I(self):
        return self.data['I']

    @property
    def Q(self):
        return self.data['Q']

    @property
    def S(self):
        '''complex signal'''
        return self.data['I'] + 1j*self.data['Q']

    @property
    def time(self):
        return self.data['time']

    @property
    def dt(self):
        return(self.data['time'].iloc[1] - self.data['time'].iloc[0])

    @property
    def max_signal(self):
        return(self.data[['I','Q']].max().max())

    @property
    def min_signal(self):
        return(self.data[['I','Q']].min().min())

    @property
    def noise_data(self):
        return(pd.concat((self.select_time_range(*self.noise_range[:2]), self.select_time_range(*self.noise_range[2:]))))

    def select_time_range(self,ta,tb):
        return(self.data[(self.data['time'] >= ta) & (self.data['time'] <= tb)].copy())

    def discriminator(self,signal,std_multiplier):
        '''A multiple of the standard deviation of the noise on signal. signal=('I','Q','IQ') '''
        return(std_multiplier * np.std(self.noise_data[signal]))

    def trim(self,t1,t2):
        '''Trims data between times t1,t2 for cleaning data (e.g. eliminating cavity ringdown)'''
        self.data = self.select_time_range(t1,t2)

    def lowpass_filter(self,order=2,cutoff=500e3):
        '''Applies digital lowpass filter to trace
        order : order of filter
        cutoff : (Hz)
        '''

        filter = sp.signal.butter(N=order,Wn=cutoff,fs=500e6,output='sos')
        for i in ['I','Q']:
            self.data[i] = sp.signal.sosfilt(filter,self.data[i])
        self.data['IQ'] = self.IQ

    def bandstop_filter(self,center,span,order=2):
        '''Applies digital lowpass filter to trace'''

        filter = sp.signal.butter(N=order,Wn=[center-span/2,center+span/2],btype='bandstop',fs=500e6,output='sos')
        for i in ['I','Q']:
            self.data[i] = sp.signal.sosfilt(filter,self.data[i])
        self.data['IQ'] = self.IQ

    def read_data(self):
        if self.data_file_type == 'pkl':
            I = pd.read_pickle(self.data_loc + self.data_name_convention + '.pkl')
            Q = pd.read_pickle(self.data_loc + self.data_name_convention.replace('I','Q') + '.pkl')
        elif self.data_file_type == 'csv':
            I = pd.read_csv(self.data_loc + self.data_name_convention + '.csv',index_col=0)
            Q = pd.read_csv(self.data_loc + self.data_name_convention.replace('I','Q') + '.csv',index_col=0)
        self._create_DataFrame(I, Q)

    def _create_DataFrame(self,I,Q):
        self.data = pd.DataFrame(columns=('time', *self.signals),dtype=np.float64)
        if (type(I) == type(Q)) and (type(I) == pd.core.series.Series):
            self.data['time'] = np.array(I.index)
            self.data['I'] = np.array(I)
            self.data['Q'] = np.array(Q)
        else:
            raise TypeError('I and Q must be supplied as pd.Series')
        self.data['IQ'] = self.IQ.fillna(0)

    def rotate(self,theta):
        '''Rotates the echo trace by theta (radians) in complex plane'''

        S = np.array(self.data['I'] + 1j*self.data['Q'])*np.exp(1j*theta)
        self.data['I'] = S.real
        self.data['Q'] = S.imag
        self.data['IQ'] = self.IQ

    def rotate_onto_I(self,rough=False):
        '''Rotates the echo so it's maximum is aligned on the I quadrature.
        rough = True : rotates max(|IQ|) onto I (Fast)
        rough = False : aligns echo by minimizing integral of Q in window defind by self.noise_range (Slow)
        '''

        theta = -1*np.angle(self.S.iloc[self.S.apply(np.abs).argmax()])

        if rough:
            self.rotate(theta)
            return theta

        def _abs_Q_int(phi):
            '''
            rotates S by phi radians, calculates integral of Q in window defined by noise range
            '''
            idx = (self.data.time[self.data.time > self.noise_range[1]].index[0],self.data.time[self.data.time < self.noise_range[2]].index[-1])
            S_rot = np.exp(1j*phi)*self.S.loc[idx[0]:idx[1]]
            a = np.abs(S_rot.apply(np.imag).sum())
            return a

        minimize_result = sp.optimize.minimize(_abs_Q_int,x0=(theta),bounds=[(theta-0.3,theta+0.3)],tol=1e-1)
        if not minimize_result.success:
            self.rotate(theta)
            self.alignment_angle = theta
            print('Polishing failed. Only rough alignment applied.')
        else:
            self.rotate(minimize_result.x[0])
            self.alignment_angle = minimize_result.x[0]
        return self.alignment_angle

    def rotate_onto_I_pca(self):
        '''Uses principle component analysis to quickly identify axis of maximum variance in echo trace, and aligns
        along I'''

        pca = sklearn.decomposition.PCA(n_components=2)
        I,Q = zip(*pca.fit_transform(self.data[['I','Q']]))
        self.data.I = I
        self.data.Q = Q
        self.data.IQ = self.IQ


    def remove_baseline(self,order=1,**kwargs):
        '''
        Removes the baseline of the data using noise_range to specify the values that should be centered at 0
        order: order of polynomial used to fit baseline. (1, 2 or 3)
        '''

        fit_classes = {1: Linear_fit, 2: Quadratic_fit, 3: Cubic_fit}
        self.baseline_fits = {i : fit_classes[order](self.noise_data['time'],self.noise_data[i],**kwargs) for i in ['I','Q']}
        for key,val in self.baseline_fits.items():
            self.data[key] = self.data[key] - np.array([val.function(i,*val.params) for i in self.data['time']])
        self.data['IQ'] = np.sqrt(self.data['I']**2 + self.data['Q']**2)

    def plot(self,save_name=None,**kwargs):
        '''Plots I, Q and IQ'''

        xlim = kwargs.get('xlim',[self.data['time'].min(),self.data['time'].max()])
        yspace = 0.05*max(np.abs(self.min_signal),np.abs(self.max_signal))
        ylim = kwargs.get('ylim',[self.min_signal - yspace,self.max_signal + yspace])
        IQ_style = kwargs.get('IQ_style','magnitude') #'complex_circle' or 'magnitude'
        axes = kwargs.get('axes',None)
        label = kwargs.get('label',None)

        if not axes:
            fig, [ax1,ax2,ax3] = generate_axes(shape=(3,1))
        else:
            ax1,ax2,ax3 = axes

        for i in zip([ax1,ax2],['I','Q'],['I (V)','Q (V)']):
            i[0].plot(self.data['time'],self.data[i[1]],label=label,lw=1)
            i[0].set_xlabel(r'Time ($\mu$s)')
            i[0].plot(xlim,[0,0],c='black',alpha=0.3)
            i[0].set_xlim(xlim)
            i[0].set_ylim(ylim)
            i[0].set_ylabel(i[2])
        if IQ_style == 'complex_circle':
            ax3.plot(self.data['I'],self.data['Q'],label=label,lw=1)
            ax3.set_xlim(ylim)
            ax3.set_ylim(ylim)
            ax3.plot(xlim,[0,0],c='black',alpha=0.3)
            ax3.plot([0,0],ylim,c='black',alpha=0.3)
            ax3.set_xlabel('I (V)')
            ax3.set_ylabel('Q (V)')
        elif IQ_style == 'magnitude':
            ax3.plot(self.data['time'],self.data['IQ'],label=label,lw=1)
            ax3.plot(xlim, [0,0], c='black', alpha=0.3)
            ax3.set_xlim(xlim)
            ax3.set_ylim([-0.05,1.05*self.data['IQ'].max()])
            ax3.set_xlabel(r'Time ($\mu$s)')
            ax3.set_ylabel('|IQ| (V)')

        if axes:
            return(ax1, ax2, ax3)
        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        else:
            plt.show()

    def fourier_transform(self,plot=True,save_name=None,**kwargs):

        self.fourier_data = pd.DataFrame(index = self.time.index, columns = ['freq','I','Q'])
        self.fourier_data['freq'] = sp.fftpack.fftfreq(self.fourier_data.shape[0],2e-9)
        self.fourier_data['I'] = sp.fftpack.fft(self.I.to_numpy())
        self.fourier_data['Q'] = sp.fftpack.fft(self.Q.to_numpy())

        if plot:
            self.plot_fourier_transform(save_name=save_name,**kwargs)

    def plot_fourier_transform(self,axes=None,save_name=None):

        _flag_axes_supplied = True
        if not axes:
            fig, axes = generate_axes(shape=(2,1))
            _flag_axes_supplied = False

        for i in zip(axes,('I','Q')):
            i[0].plot(self.fourier_data['freq']*1E-3,self.fourier_data[i[1]].apply(np.abs))
            i[0].set_xlabel('Fourier Frequency (KHz)')
            i[0].set_ylabel('{} Fourier Amplitude'.format(i[1]))
            i[0].set_xlim([0,1000])

        if _flag_axes_supplied:
            return axes
        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        else:
            plt.show()

    def integrate_echo_with_discriminators(self,std_multiplier=1,**kwargs):
        '''
        Integrates the echo signals by filtering summing the data in self.data that is filtered based on discriminators
        std_multiplier: the multiplier of the std. dev. of the noise used for constructing discriminators
        '''

        discriminators = {i : self.discriminator(i,std_multiplier) for i in self.signals}
        _I = self.data[(self.data['I'] > discriminators['I']) | (self.data['I'] < -1*discriminators['I'])]['I']
        _Q = self.data[(self.data['Q'] > discriminators['Q']) | (self.data['Q'] < -1*discriminators['Q'])]['Q']
        _IQ = self.data[self.data['IQ'] > discriminators['IQ']]['IQ']

        self.integrated_echo = {}
        self.integrated_echo['I'] = _I.sum() * self.dt
        self.integrated_echo['Q'] = _Q.sum() * self.dt
        self.integrated_echo['IQ'] = (_IQ - discriminators['IQ']).sum() * self.dt
        self.integrated_echo['|I|'] = np.abs(_I).sum()*self.dt
        self.integrated_echo['|Q|'] = np.abs(_Q).sum()*self.dt

        self.integrated_echo_uncertainty = {}
        for i in zip([_I,_Q,_IQ],self.signals):
            self.integrated_echo_uncertainty[i[1]] = i[0].count()*discriminators[i[1]]*self.dt #/std_multiplier
        for i in zip(['I','Q'],['|I|','|Q|']):
            self.integrated_echo_uncertainty[i[1]] = self.integrated_echo_uncertainty[i[0]]

    def integrate_echo(self,**kwargs):
        '''
        Integrates the echo signals by a simple sum of the signal in the time between the two slices designated as noise via noise_range
        '''

        self.integrated_echo = {}
        _data = self.data[(self.data['time'] > self.noise_range[1]) & (self.data['time'] < self.noise_range[2])]
        self.integrated_echo['I'] = _data['I'].sum() * self.dt
        self.integrated_echo['Q'] = _data['Q'].sum() * self.dt
        self.integrated_echo['IQ'] = _data['IQ'].sum() * self.dt
        self.integrated_echo['|I|'] = _data['I'].apply(np.abs).sum() * self.dt
        self.integrated_echo['|Q|'] = _data['Q'].apply(np.abs).sum() * self.dt

    def compare_with_trace(self,trace,save_name=None,**kwargs):
        '''
        Overlays Echo_trace.plot of the two Echo_trace objects for a direct comparison
        '''

        labels = kwargs.get('labels',(None,None))
        axes = kwargs.get('axes',None)

        _flag_axis_supplied = True
        if not axes:
            fig,axes = generate_axes(shape=(3,1))
            _flag_axis_supplied = False

        self.plot(axes=axes,label=labels[0])
        trace.plot(axes=axes,label=labels[1])

        def create_ylim(signal):
            _min = min(self.data[signal].min(),trace.data[signal].min())
            _max = max(self.data[signal].max(),trace.data[signal].max())
            buffer = 0.1*max(_max,-1*_min)
            return(_min-buffer,_max+buffer)

        for i in zip(axes,self.signals):
            i[0].set_ylim(create_ylim(i[1]))
            if labels[0]:
                i[0].legend()

        if _flag_axis_supplied:
            return(axes)
        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        else:
            plt.show()

    def overlay_traces(self,traces,save_name=None,**kwargs):
        '''
        Overlays Echo_trace.plot of the two Echo_trace objects for a direct comparison
        '''

        labels = kwargs.get('labels',[None for i in range(len(traces) + 1)])

        axes = kwargs.get('axes',None)
        _flag_axis_supplied = True
        if not axes:
            fig,axes = generate_axes(shape=(3,1),figsize=(6,5))
            _flag_axis_supplied = False

        for i in zip(axes,self.signals):
            i[0].fill_between(self.data['time'],self.data[i[1]],0,alpha=0.5,label=labels[0])
            # i[0].set_xlabel(r'Time ($\mu$s)')
            i[0].set_xticks([])
            i[0].set_ylabel('{} (V)'.format(i[1]))
            for n,j in enumerate(traces):
                i[0].fill_between(j.data['time']-(n+1)*15,j.data[i[1]],0,alpha=0.5,label=labels[n+1])

        if labels[0]:
            for i in axes:
                i.legend(loc=2)

        if _flag_axis_supplied:
            return(axes)
        plt.tight_layout()
        if save_name:
            plt.savefig(self.save_loc + save_name)
            plt.close()
        else:
            plt.show()
