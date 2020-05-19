import pandas as pd
import numpy as np
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
import warnings

def generate_axes(shape,**kwargs):

    '''
    shape : tuple (num rows, num columns)
    '''

    figsize = kwargs.get('figsize', (4*shape[1], 3*shape[0]))

    fig, axes = plt.figure(figsize=figsize), []
    for j in range(shape[0]):
        for i in range(shape[1]):
            axes.append(plt.subplot2grid(shape=shape, loc=(j, i)))
    return(fig,axes)

def remove_polynomial_baseline(x, y, x1, x2, x3, x4, order=1):
    '''
    A function for fitting the baseline of an echo.
    x = np array
    x1,x2,x3,x4 = values within x corresponding to the baseline
    y = np array
    order = order of polynomial (2 or 3)
    '''

    if order == 1:
        poly = lambda x, a, b: a + b*x
    if order == 2:
        poly = lambda x, a, b, c: a + b*x + c*x**2
    if order == 3:
        poly = lambda x, a, b, c, d: a + b*x + c*x**2 + d*x**3

    data = pd.DataFrame.from_dict({'x': x, 'y': y})
    _cut1 = data[(data['x'] >= x1) & (data['x'] <= x2)]
    _cut2 = data[(data['x'] >= x3) & (data['x'] <= x4)]
    rdata = pd.concat((_cut1, _cut2))

    _x = np.array(rdata['x'])
    _y = np.array(rdata['y'])
    popt, pcov = sp.optimize.curve_fit(poly, _x, _y)
    fit = np.array([poly(x, *popt)])

    return (np.subtract(y, fit)[0])

class circle():

    '''A circle in the complex plane. Used in plotting functions'''

    def __init__(self,r,x0=0,y0=0):
        
        self.r = r
        self.x0 = x0
        self.y0 = y0
        self.create_coords()
        
    def create_coords(self):
        
        theta = np.linspace(0,2*np.pi,100)
        self.coords = (self.x0 + 1j*self.y0) + self.r*np.exp(1j*theta)

class Echo_experiment():

    '''Base class for experiments.'''

    def __init__(self,data_loc,save_loc=None,**kwargs):
        self.data_loc = data_loc
        self.save_loc = save_loc

    def read_data(self,**kwargs):

        if self.data_file_type == 'pkl':
            self.Is = pd.read_pickle(self.data_loc + self.data_name_convention + '.pkl')
            self.Qs = pd.read_pickle(self.data_loc + self.data_name_convention.replace('I','Q') + '.pkl')
        elif self.data_file_type == 'csv':
            self.Is = pd.read_csv(self.data_loc + self.data_name_convention + '.csv',index_col=0)
            self.Qs = pd.read_csv(self.data_loc + self.data_name_convention.replace('I','Q') + '.csv',index_col=0)

        self.time = np.array(self.Is.index)
        self.columns = np.array(self.Is.columns)

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
        Q_lims = kwargs.get('I_lims', [-0.5, 0.5])
        IQ_lims = kwargs.get('IQ_lims',[-0.05,1.0])
        IQ_style = kwargs.get('IQ_style','complex_circle') #'complex_circle' or 'magnitude'
        axes = kwargs.get('axes',None) #user can supply ax1,ax2,ax3 which will be returned to them

        if not axes:
            fig, [ax1,ax2,ax3] = generate_axes(shape=(3,1))
        else:
            ax1,ax2,ax3 = axes

        for i in zip([ax1,ax2],['I','Q'],[I_lims,Q_lims],['I (V)','Q (V)']):
            i[0].plot(self.data['time'],self.data[i[1]])
            i[0].set_xlabel('Time (us)')
            i[0].plot(min_max_times,[0,0],c='black',alpha=0.3)
            i[0].set_xlim(min_max_times)
            i[0].set_ylim(i[2])
            i[0].set_ylabel(i[3])

        if IQ_style == 'complex_circle':
            ax3.plot(self.data['I'],self.data['Q'])
            ax3.set_xlim(I_lims)
            ax3.set_ylim(Q_lims)
            ax3.plot(I_lims,[0,0],c='black',alpha=0.3)
            ax3.plot([0,0],Q_lims,c='black',alpha=0.3)
            ax3.set_xlabel('I (V)')
            ax3.set_ylabel('Q (V)')
        elif IQ_style == 'magnitude':
            ax3.plot(self.data['time'],self.data['IQ'])
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

        # self.IQ_integrated_echo = _IQ.sum()*self.dt
        # self.I_integrated_echo = _I.sum()*self.dt
        # self.Q_integrated_echo = _Q.sum()*self.dt

class Sweep_experiment(Echo_experiment):

    '''
    A class for experiments where IQ traces are collected as a function of some 1D sweep parameter (e.g. pulse power)
    Provides simple way to trim the data, subtract the baseline on I and Q, integrate the echos, and plot the data
    '''

    def __init__(self,data_loc,save_loc=None,**kwargs):

        super().__init__(data_loc,save_loc)
        self.data_name_convention = kwargs.get('data_name_convention', 'Is')
        self.data_file_type = kwargs.get('data_file_type', 'pkl')
        self.read_data()
        self.sweep_parameter = kwargs.get('sweep_parameter', None)

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

        fig, [ax1,ax2,ax3] = generate_axes(shape=(3,1))
        extent = [float(self.columns[0]),float(self.columns[-1]),self.time[0],self.time[-1]]
        IQmags = (self.Is**2 + self.Qs**2).apply(np.sqrt)

        for i in zip([ax1,ax2,ax3],[self.Is,self.Qs,IQmags],['I','Q','|IQ|']):
            im = i[0].imshow(i[1],aspect='auto',origin='lower',extent=extent)
            fig.colorbar(im,ax=i[0],shrink=0.8)
            i[0].set_ylabel('Time (us)')
            i[0].set_xlabel(self.sweep_parameter)

        return_fig = kwargs.get('return_fig',False)
        if return_fig:
            return(fig,(ax1,ax2,ax3))

        plt.tight_layout()
        save_name = kwargs.get('save_name',None)
        if save_name:
            plt.savefig(self.save_loc + save_name)
        plt.show()

    def remove_baseline(self,t1,t2,t3,t4,order=1,**kwargs):
        ''''
        Remove baseline from data. Updates self.Is and self.Qs
        t1 - t4: define two regions of baseline on either side of the echo
        order: order of polynomial used in fitting of baseline, defaults to 1 = linear fit
        '''

        self.Is_raw = self.Is
        self.Qs_raw = self.Qs

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
                # i[0].errorbar(x,self.integrated_echos.loc[:,i[1]],self.integrated_echo_uncertainties.loc[:,i[1]]/2,linestyle="None",fmt='o',markersize=3)
                _yplus = np.array(self.integrated_echos.loc[:,i[1]]+self.integrated_echo_uncertainties.loc[:,i[1]]/2)
                _yminus = np.array(self.integrated_echos.loc[:,i[1]]-self.integrated_echo_uncertainties.loc[:,i[1]]/2)
                i[0].fill_between(x,_yplus,_yminus,color='b',alpha=0.2)
                i[0].set_ylabel(i[1])
                i[0].set_xlabel(self.sweep_parameter)

            plt.tight_layout()
            save_name = kwargs.get('save_name',None)
            if save_name:
                plt.savefig(self.save_loc + save_name)
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









