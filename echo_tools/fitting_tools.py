import numpy as np
import scipy as sp
from scipy import optimize



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

    def print_result(self,**kwargs):
        if len(self.symbols_list) == len(self.params):
            print('Fit Result')
            print(self.descriptor)
            for i in zip(self.symbols_list,self.params):
                print(i[0] + ' : ' + str(i[1]))
        else:
            print('Length of symbols_list does not match params. Fit not performed or error in class definition.')

class Fit_1D(Fit_base):

    def __init__(self,x,y):
        super().__init__()
        self.x = x
        self.y = y

    def fit(self,guess=None):

        try:
            self.params,self.covariance = sp.optimize.curve_fit(self.function,x,y,p0=guess)
            self.fit = np.array([self.function(i,*self.params) for i in self.x])
        except:
            self._flag_fit_error = True




class T1_fit(Fit_1D):

    def __init__(self,time,value,**kwargs):
        super().__init__(x=time,y=value)
        self.function = lambda x,T1,a,b: a + b * np.exp(-x/T1)
        self.descriptor = r'$a+b\exp{(-t/T_1)}$'
        self.symbols_list = ['a,b,T1']

        self.guess = kwargs.get('guess',None)
        self.fit(self.guess)
        self.print_result()

