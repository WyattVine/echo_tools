import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def remove_linear_baseline(x,y,x1,x2,x3,x4):
    
    '''
    x = np array
    x1,x2,x3,x4 = values within x corresponding to the baseline
    y = np array
    '''
    
    linear_func = lambda x,a,b: a + b*x
    
    data = pd.DataFrame.from_dict({'x':x,'y':y})
    _cut1 = data[(data['x'] >= x1) & (data['x'] <= x2)]
    _cut2 = data[(data['x'] >= x3) & (data['x'] <= x4)]
    rdata = pd.concat((_cut1,_cut2))
    
    _x = np.array(rdata['x'])
    _y = np.array(rdata['y'])
    popt,pcov = sp.optimize.curve_fit(linear_func,_x,_y)
    fit = np.array([linear_func(x,*popt)])
    
    return(np.subtract(y,fit)[0])


class circle():
    
    def __init__(self,r,x0=0,y0=0):
        
        self.r = r
        self.x0 = x0
        self.y0 = y0
        self.create_coords()
        
    def create_coords(self):
        
        theta = np.linspace(0,2*np.pi,100)
        self.coords = (self.x0 + 1j*self.y0) + self.r*np.exp(1j*theta)
        

class Echo_trace():
    
    '''
    Basic representation of a single echo time trace.
    '''
    
    def __init__(self,time,I,Q,**kwargs):
        
        '''
        time = np.array
        I = np.array
        Q = np.array
        '''
        
        self.data = pd.DataFrame(columns=('time','I','Q','S','IQ'))
        self.data['time'] = time
        self.data['I'] = I
        self.data['Q'] = Q
        self.data['S'] = I + 1j*Q
        self.data['IQ'] = np.abs(self.data['S'])
        self.dt = time[1] - time[0]
        
        self._discriminator_flag = False #True when discriminators have been created
        
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
        Q_lims = kwargs.get('I_lims', [-0.5, 0.5])

        shape = (3,1)
        plt.figure(figsize=(4,12))
        ax1 = plt.subplot2grid(shape=shape,loc=(0,0))
        ax2 = plt.subplot2grid(shape=shape,loc=(1,0))
        ax3 = plt.subplot2grid(shape=shape,loc=(2,0),rowspan=1)
        
        ax1.plot(self.data['time'],self.data['I'])
        ax2.plot(self.data['time'],self.data['Q'])
        for i in [ax1,ax2]:
            i.set_xlabel('Time (us)')
            i.plot(min_max_times,[0,0],c='black',alpha=0.3)
            i.set_xlim(min_max_times)
        ax1.set_ylim(I_lims)
        ax2.set_ylim(Q_lims)
        ax1.set_ylabel('I (V)')
        ax2.set_ylabel('Q (V)')

        ax3.plot(self.data['I'],self.data['Q'])
        ax3.set_xlim(I_lims)
        ax3.set_ylim(Q_lims)
        ax3.plot(I_lims,[0,0],c='black',alpha=0.3)
        ax3.plot([0,0],Q_lims,c='black',alpha=0.3)
        ax3.set_xlabel('I (V)')
        ax3.set_ylabel('Q (V)')

        if self._discriminator_flag:
            discriminator_circle = circle(r=self.discriminators['IQ'])
            ax3.plot(discriminator_circle.coords.real,discriminator_circle.coords.imag,c='r')
            ax1.fill_between(min_max_times,[self.discriminators['I'],self.discriminators['I']],[-1*self.discriminators['I'],-1*self.discriminators['I']],color='r',alpha=0.2)
            ax2.fill_between(min_max_times,[self.discriminators['Q'],self.discriminators['Q']],[-1*self.discriminators['Q'],-1*self.discriminators['Q']],color='r',alpha=0.2)

        plt.tight_layout()
        plt.show()
        
    def create_discriminators(self,t1,t2,t3=None,t4=None):
        '''
        Using t1 - t4 it creates single values corresponding to the noise in I, Q, and IQ that can be used for
        slicing self.data
        '''
        
        _generate_reduced = lambda ta,tb:  self.data[(self.data['time'] >= ta) & (self.data['time'] <= tb)]
        _reduced = _generate_reduced(t1,t2)
        if t3:
            _reduced = pd.concat((_reduced,_generate_reduced(t3,t4)))
            
        self.discriminators = {}
        for i in ['I','Q']:
            vals = np.array(_reduced[i])
            self.discriminators[i] = max(vals.max(),-1*vals.min())
        self.discriminators['IQ'] = np.array(_reduced['IQ']).max()

        self._discriminator_flag = True
        
    def integrate_echo(self,**kwargs):

        noise_range = kwargs.get('noise_range',None) #tuple of t1,t2,t3,t4
        if noise_range:
            self.create_discriminators(*noise_range)

        _IQ = self.data[self.data['IQ'] > self.discriminators['IQ']]['IQ']
        _I = self.data[(self.data['I'] > self.discriminators['I']) | (self.data['I'] < -1*self.discriminators['I'])]['I']
        _Q = self.data[(self.data['Q'] > self.discriminators['Q']) | (self.data['Q'] < -1*self.discriminators['Q'])]['Q']
        
        self.integrated_echo = {}
        self.integrated_echo['IQ'] = (_IQ - self.discriminators['IQ']).sum()*self.dt
        self.integrated_echo['I'] = (np.abs(_I)-self.discriminators['I']).sum()*self.dt
        self.integrated_echo['Q'] = (np.abs(_Q)-self.discriminators['Q']).sum()*self.dt
        
        # self.IQ_integrated_echo = _IQ.sum()*self.dt
        # self.I_integrated_echo = _I.sum()*self.dt
        # self.Q_integrated_echo = _Q.sum()*self.dt

class Sweep_experiment():

    def __init__(self,Is,Qs,**kwargs):

        self.Is = Is #pd DataFrame
        self.Qs = Qs #pd DataFrame
        self.time = np.array(Is.index)
        self.columns = np.array(Is.columns)
        self.sweep_parameter = kwargs.get('sweep_parameter',None)
        self.save_loc = None

    def trim(self,t1,t2):
        '''
        trims Is and Qs to only include times between t1 and t2 (e.g. to cut out ringdown)
        '''

        self.Is = self.Is.loc[t1:t2,:]
        self.Qs = self.Qs.loc[t1:t2, :]
        self.time = np.array(self.Is.index)


    def plot_2D(self,save_name=None,**kwargs):

        shape = (3, 1)
        fig = plt.figure(figsize=kwargs.get('figsize',(4,12)))
        ax1 = plt.subplot2grid(shape=shape, loc=(0, 0))
        ax2 = plt.subplot2grid(shape=shape, loc=(1, 0))
        ax3 = plt.subplot2grid(shape=shape, loc=(2, 0))

        extent = [float(self.columns[0]),float(self.columns[-1]),self.time[0],self.time[-1]]
        IQmags = np.sqrt(self.Is**2 + self.Qs**2)

        for i in zip([ax1,ax2,ax3],[self.Is,self.Qs,IQmags],['I','Q','|IQ|']):
            im = i[0].imshow(i[1],aspect='auto',origin='lower',extent=extent)
            fig.colorbar(im,ax=i[0],shrink=0.8)
            i[0].set_ylabel('Time (us)')
            i[0].set_xlabel(self.sweep_parameter)

        return_fig = kwargs.get('return_fig',False)
        if return_fig:
            return(fig,(ax1,ax2,ax3))

        if save_name:
            plt.savefig(self.save_loc + save_name)
        plt.show()


    def remove_baseline(self,t1,t2,t3,t4):

        Is_corr = pd.DataFrame(index=self.Is.index,columns=self.Is.columns)
        Qs_corr = pd.DataFrame(index=self.Qs.index, columns=self.Qs.columns)
        for col in np.array(self.Is.columns):
            Is_corr.at[:,col] = remove_linear_baseline(self.time,np.array(Is.loc[:,col]),t1,t2,t3,t4)
            Qs_corr.at[:, col] = remove_linear_baseline(self.time, np.array(Qs.loc[:, col]),t1,t2,t3,t4)

        self.Is = Is_corr
        self.Qs = Qs_corr

    def integrate_echos(self,noise_range,plot=True,**kwargs):
        ''''
        noise_range = (t1,t2,t3,t4)
        '''

        self.integrated_echos = pd.DataFrame(index = self.columns, columns=('I','Q','IQ'))
        self.discriminators = pd.DataFrame(index=self.columns, columns=('I','Q','IQ'))
        for i in self.columns:
            _I = np.array(self.Is.loc[:,i])
            _Q = np.array(self.Qs.loc[:,i])
            S = Echo_trace(self.time,_I,_Q)
            S.integrate_echo(noise_range=noise_range)
            for col in ['I','Q','IQ']:
                self.integrated_echos.loc[i,col] = S.integrated_echo[col]
                self.discriminators.loc[i, col] = S.discriminators[col]


        if plot:

            x = [float(i) for i in self.integrated_echos.index]
            shape = (3, 1)
            fig = plt.figure(figsize=(4, 9))
            ax1 = plt.subplot2grid(shape=shape, loc=(0, 0))
            ax2 = plt.subplot2grid(shape=shape, loc=(1, 0))
            ax3 = plt.subplot2grid(shape=shape, loc=(2, 0))

            for i in zip([ax1, ax2, ax3], ['I', 'Q', 'IQ']):
                i[0].scatter(x,self.integrated_echos.loc[:,i[1]])
                i[0].set_ylabel(i[1])
                i[0].set_xlabel(self.sweep_parameter)

            plt.tight_layout()
            plt.show()







