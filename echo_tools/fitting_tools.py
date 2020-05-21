import numpy as np
import scipy as sp
from scipy import optimize
import matplotlib.pyplot as plt
from .utilities import *
update_matplot_style()


def data_type_checker(x,y):
    '''
    Ensures proper format of x and y for fitting classes
    '''

    if (type(x) == type(y)) and (type(x) == np.ndarray):
        return(x, y)
    elif x.empty:
        return(np.array(y.index),np.array(y))
    elif (type(x) == type(y)) and (type(x) == pd.core.series.Series):
        return(np.array(x),np.array(y))
    else:
        raise ValueError('x and y must be of type np.ndarrays or pd.Series')

class Fit_base():

    '''
    Base class for fitting.
    '''

    def __init__(self):

        self.function = None #the function being fit
        self.descriptor = None #a latex compatible string describing the function being fit, e.g. r'$a+b\times x$
        self.symbols_list = None #a list of strings for the symbols in descriptor, e.g. ['a','b']
        self._flag_fit_performed = False
        self._flag_fit_error = False
        self.fit = None #np.ndarray storing the fit generated
        self.params = None #stores the parameters that have been fit
        self.covariance = None


    def result_string(self):
        '''
        Creates a single string for conveniently displaying results of a fit in a plot
        '''

        if len(self.symbols_list) != len(self.params):
            print('Length of symbols_list does not match params. Fit not performed or error in class definition.')
            return

        s = self.descriptor + '\n'
        for i in zip(self.symbols_list,self.params):
            s += i[0] + ' : {:.2e}'.format(i[1]) + '\n'
        return(s.rstrip())

    def print_result(self,**kwargs):
        '''
        Prints the result of the fit to terminal
        '''

        print('Fit Result')
        for i in self.result_string().split('\n'):
            print(i)

class Fit_1D(Fit_base):
    '''
    Class for fitting 1D data
    '''

    def __init__(self,x,y,**kwargs):
        super().__init__()
        self.x = x #np.ndarray
        self.y = y #np.ndarray

    def perform_fit(self,guess=None):

        try:
            self.params, self.covariance = sp.optimize.curve_fit(self.function, self.x, self.y, p0=guess)
            self.fit = np.array([self.function(i, *self.params) for i in self.x])
        except:
            self._flag_fit_error = True
            print('Error in fitting')

        self._flag_fit_performed = True

    def plot_result(self):
        plt.plot(self.x,self.y,'o')
        plt.plot(self.x,self.fit)
        plt.show()

class Linear_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        x,y = data_type_checker(x,y)
        super().__init__(x,y)
        self.function = lambda x,a,b: a + b*x
        self.descriptor = r'$a+b\times x$'
        self.symbols_list = ['a','b']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (self.y[0],(self.y[-1] - self.y[0])/(self.x[-1] - self.x[0]))
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Quadratic_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        x, y = data_type_checker(x, y)
        super().__init__(x, y)
        self.function = lambda x, a, b, c, d: a + b*(x - c) + d*(x - c)**2
        self.descriptor = r'$a+b\times x$'
        self.symbols_list = ['a', 'b', 'c', 'd']

        self.guess = kwargs.get('guess', None)
        self.perform_fit(guess=self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result', False):
                self.print_result()
            if kwargs.get('plot_result', False):
                self.plot_result()

class Cubic_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        x, y = data_type_checker(x, y)
        super().__init__(x, y)
        self.function = lambda x, a, b, c, d, e: a + b * (x - c) + d * (x - c) ** 2 + e*(x-c)**2
        self.descriptor = r'$a+b\times x$'
        self.symbols_list = ['a', 'b', 'c', 'd','e']

        self.guess = kwargs.get('guess', None)
        self.perform_fit(guess=self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result', False):
                self.print_result()
            if kwargs.get('plot_result', False):
                self.plot_result()

class Exponential_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        x,y = data_type_checker(x,y)
        super().__init__(x,y)
        self.function = lambda x,a,b,c: a + b * np.exp(-x/c)
        self.descriptor = r'$a+b\times \exp{(x/c)}$'
        self.symbols_list = ['a','b','c']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (self.y[0],self.y[-1]-self.y[0],x[len(x)//3])
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class T1_fit(Exponential_fit):

    def __init__(self,value,time=pd.Series(dtype=float),**kwargs):
        super().__init__(time,value,**kwargs)

        self.descriptor = r'$a+b\times \exp{(-t/T_1)}$'
        self.symbols_list = ['a','b','T1']


