import numpy as np
import scipy as sp
from scipy import optimize
import matplotlib.pyplot as plt
from .utilities import *
update_matplot_style()

def data_type_checker(x,y):

    if (type(x) == type(y)) and (type(x) == np.ndarray):
        return(x, y)

    elif x.empty:
        return(np.array(y.index),np.array(y))

    else:
        raise ValueError('x and y must be of type np.ndarrays or pd.Series')




class Fit_base():

    def __init__(self):

        self.function = None
        self.descriptor = None
        self._flag_fit_performed = False
        self._flag_fit_error = False
        self.fit = None
        self.params = None
        self.covariance = None
        self.symbols_list = None

    def result_string(self):

        if len(self.symbols_list) != len(self.params):
            print('Length of symbols_list does not match params. Fit not performed or error in class definition.')
            return

        s = self.descriptor + '\n'
        for i in zip(self.symbols_list,self.params):
            s += i[0] + ' : {:.2e}'.format(i[1]) + '\n'
        return(s.rstrip())

    def print_result(self,**kwargs):
        print('Fit Result')
        for i in self.result_string().split('\n'):
            print(i)

class Fit_1D(Fit_base):

    def __init__(self,x,y):
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




class T1_fit(Fit_1D):

    def __init__(self,value,time=pd.Series(dtype=float),**kwargs):

        x,y = data_type_checker(time,value)
        super().__init__(x,y)
        self.function = lambda x,T1,a,b: a + b * np.exp(-x/T1)
        self.descriptor = r'$a+b\exp{(-t/T_1)}$'
        self.symbols_list = ['T1','a','b']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (x[len(x)//3], self.y[0], self.y[-1]-self.y[0])
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

