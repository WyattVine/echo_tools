import numpy as np
import scipy as sp
from scipy import optimize
import matplotlib.pyplot as plt
from .utilities import *
update_matplot_style()





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
        self.covariance = None #covariance of fit

    @property
    def std_dev(self):
        return(np.sqrt(np.diag(self.covariance)))

    @property
    def result_string(self):
        '''Creates a single string for conveniently displaying results of a fit in a plot'''

        if len(self.symbols_list) != len(self.params):
            print('Length of symbols_list does not match params. Fit not performed or error in class definition.')

        s = self.descriptor + '\n'
        for i in zip(self.symbols_list,self.params,self.std_dev):
            s += i[0] + r' : ${:.1e} \pm {:.0e}$'.format(i[1],i[2]) + '\n'
        return(s.rstrip())

    def print_result(self):
        print('Fit Result')
        for i in self.result_string.split('\n'):
            print(i)


class Fit_1D(Fit_base):

    '''
    Class for fitting 1D data
    '''

    def __init__(self,x,y,**kwargs):
        super().__init__()
        self.x = x
        self.y = y
        self.data_type_checker

    @property
    def data_type_checker(self):
        '''Ensures proper format of x and y'''
        if (type(self.x) == type(self.y)) and (type(self.x) == np.ndarray):
            return
        elif self.x.empty:
            self.x = np.array(self.y.index)
            self.y = np.array(self.y)
        elif (type(self.x) == type(self.y)) and (type(self.x) == pd.core.series.Series):
            self.x = np.array(self.x)
            self.y = np.array(self.y)
        elif (type(self.x) == np.ndarray) and (type(self.y) == pd.core.series.Series):
            self.y = np.array(self.y)
        else:
            raise ValueError('x and y must be of type np.ndarrays or pd.Series')

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

        super().__init__(x,y)
        self.function = lambda x,a,b: a + b*x
        self.descriptor = r'$a+b\times x$'
        self.symbols_list = [r'$a$',r'$b$']

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

        super().__init__(x, y)
        self.function = lambda x, a, b, c: a + b*(x - c)**2
        self.descriptor = r'$a + b \times (x-c)^2$'
        self.symbols_list = [r'$a$', r'$b$', r'$c$']

        self.guess = kwargs.get('guess', None)
        self.perform_fit(guess=self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result', False):
                self.print_result()
            if kwargs.get('plot_result', False):
                self.plot_result()

class Poly2_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x, y)
        self.function = lambda x, a, b, c, d: a + b*(x - c) + d*(x - c)**2
        self.descriptor = r'$a+b\times (x-c) + d\times(x-c)**2$'
        self.symbols_list = [r'$a$', r'$b$', r'$c$', r'$d$']

        self.guess = kwargs.get('guess', None)
        self.perform_fit(guess=self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result', False):
                self.print_result()
            if kwargs.get('plot_result', False):
                self.plot_result()

class Poly3_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x, y)
        self.function = lambda x, a, b, c, d, e: a + b * (x - c) + d * (x - c) ** 2 + e*(x-c)**2
        self.descriptor = r'$a+b\times x$'
        self.symbols_list = [r'$a$', r'$b$', r'$c$', r'$d$',r'$e$']

        self.guess = kwargs.get('guess', None)
        self.perform_fit(guess=self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result', False):
                self.print_result()
            if kwargs.get('plot_result', False):
                self.plot_result()


class Exponential_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,b,c: a + b * np.exp(-x/c)
        self.descriptor = r'$a+b\times \exp{(-x/c)}$'
        self.symbols_list = [r'$a$',r'$b$',r'$c$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (self.y[0],self.y[-1]-self.y[0],self.x[len(self.x)//3])
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Exponential_fit2(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,b: a*(1 - np.exp(-x/b))
        self.descriptor = r'$a \times (1 - \exp{(x/a)})$'
        self.symbols_list = [r'$a$',r'$b$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (self.y.max(),self.x[len(self.x)//3])
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Exponential_fit3(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,b,c,d: a*np.exp(-(x-b)/c) + d
        self.descriptor = r'$a \times \exp{-(x-b)/c} + d$'
        self.symbols_list = [r'$a$',r'$b$',r'$c$',r'$d$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (self.y.ptp(),self.x[0],self.x[len(self.x)//3] - self.x[0],self.y.min())
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Exponential_fit4(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,c,d,e: a*np.exp(-(x/c)**d) + e
        self.descriptor = r'$a \times \exp{-(x/c)**d} + e$'
        self.symbols_list = [r'$a$',r'$c$',r'$d$',r'$e$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (self.y.ptp(),self.x[0],self.x[len(self.x)//3] - self.x[0],1,self.y.min())
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Exponential_fit5(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,b: np.exp(-(x/a)**b)
        self.descriptor = r'$\exp{-(x/a)**b}$'
        self.symbols_list = [r'$a$',r'$b$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (self.x[len(self.x)//3] - self.x[0],2)
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Exponential_fit6(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,b: a*np.exp(-x/b)
        self.descriptor = r'$a*\exp{-x/a}$'
        self.symbols_list = [r'$a$',r'$b$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (1,self.x[len(self.x)//3] - self.x[0])
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Exponential_fit7(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a: np.exp(-x/a)
        self.descriptor = r'$\exp{-x/a}$'
        self.symbols_list = [r'$a$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (self.x[len(self.x)//3] - self.x[0])
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Richards_curve_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,b,c,d: a*(1 + d*np.exp(-b*(x-c)))**(-1/d)
        self.descriptor = r'$a \times [1 + d\exp{(-b*(x-c))}]^(-1/d)$'
        self.symbols_list = [r'$a$',r'$b$',r'$c$',r'$d$',]

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (self.y.max(),0.1,self.x[len(self.x)//5],1)
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Double_Exponential_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,b,c,d,e: a + b * np.exp(-x/c) + d * np.exp(-x/e)
        self.descriptor = r'$a+b\times\exp{(x/c)}+d\times\exp{(x/e)}$'
        self.symbols_list = [r'$a$',r'$b$',r'$c$',r'$d$',r'$e$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (self.y[0],self.y[-1]-self.y[0],self.x[len(self.x)//3],self.y[-1]-self.y[0],self.x[len(self.x)//2])
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Sine_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,b: a * np.sin(x - b)
        self.descriptor = r'$a\times \sin(x - b)$'
        self.symbols_list = [r'$a$',r'$b$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (max(self.y),0)
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Sine_fit2(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a: a * np.sin(x)
        self.descriptor = r'$a\times \sin(x)$'
        self.symbols_list = [r'$a$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (max(self.y))
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Sine_fit3(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,b: a * np.sin(x) + b
        self.descriptor = r'$a\times \sin(x) + b$'
        self.symbols_list = [r'$a$',r'$b$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (max(self.y)-min(self.y),min(self.y))
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Sine_normalized_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a: np.sin(x - a)
        self.descriptor = r'$\sin(x - a)$'
        self.symbols_list = [r'$a$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (0)
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Sine_abs_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,b: a*np.abs(np.sin((x - b + np.pi)/2))
        self.descriptor = r'$a \times |\sin((x - b + \pi)/2)|$'
        self.symbols_list = [r'$a$',r'$b$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (y.max(),0)
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Sine_abs_fit2(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,b: a*np.abs(np.sin(x) + b)
        self.descriptor = r'$a \times |\sin(x)| + b$'
        self.symbols_list = [r'$a$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (y.max(),0)
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Maccas_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,b,c: a + b*20*np.log10(np.abs(np.sin((x - c)/2)))
        self.descriptor = r'$20 \times \log_{10}(a \times |\sin((x - b)/2)|)$'
        self.symbols_list = [r'$a$',r'$b$',r'$c$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (0,1,0)
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Maccas_fit2(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,b,c: a + b*np.abs(np.sin((x - c)/2))
        self.descriptor = r'a + b \times |\sin((x - c)/2)|)$'
        self.symbols_list = [r'$a$',r'$b$',r'$c$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (0,1,0)
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Maccas_fit3(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,b: a + b*np.abs(np.sin((x - np.pi)/2))
        self.descriptor = r'$a + b \times |\sin((x - \pi)/2)|$'
        self.symbols_list = [r'$a$',r'$b$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (0,1)
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class T1_fit(Exponential_fit):

    '''Note that x and y are backwards with respect to most classes.
    allows you to just pass in dataframe with integrated echos as value'''
    def __init__(self,value,time=pd.Series(dtype=float),**kwargs):
        super().__init__(time,value,**kwargs)

        self.descriptor = r'$a+b\times \exp{(-t/T_1)}$'
        self.symbols_list = [r'$a$',r'$b$',r'$T_1$']

class Rayleigh_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)

        a = y.sum() * (x[1] - x[0]) #a is a normalization constant and is not fit.

        self.function = lambda x,sig: a * x/sig**2 * np.exp(-0.5*(x/sig)**2)
        self.descriptor = r'$a \times x/\sigma^2 \times \exp{(-(x/\sigma)^2/2)}$'
        self.symbols_list = [r'$\sigma$',r'$a$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (self.x.max()/10)
        self.perform_fit(guess = self.guess)

        '''update params and covariance with normalization constant for full features'''
        self.params = np.append(self.params,a)
        self.covariance = np.append(self.covariance,[0,0,0]).reshape(2,2)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Gaussian_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)

        a = y.sum() * (x[1] - x[0]) #a is a normalization constant and is not fit.

        self.function = lambda x,mu,sig: a/(sig*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sig)**2)
        self.descriptor = r'$a/(\sigma * \sqrt{2*\pi}) \times \exp{(-((x-\mu)/\sigma)^2/2)}$'
        self.symbols_list = [r'$\mu$',r'$\sigma$',r'$a$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (self.x[self.y.argmax()],np.ptp(self.x)/5)
        self.perform_fit(guess = self.guess)

        '''update params and covariance with normalization constant for full features'''
        self.params = np.append(self.params,a)
        c = np.zeros(9).reshape(3,3)
        c[:2,:2] = self.covariance
        self.covariance = c

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()

class Lorentzian_fit(Fit_1D):

    def __init__(self,x,y,**kwargs):

        super().__init__(x,y)
        self.function = lambda x,a,b,c: a * (b / ((x-c)**2 + b**2))
        self.descriptor = r'test'
        self.symbols_list = [r'$a$',r'$b$',r'$c$']

        self.guess = kwargs.get('guess',None)
        if not self.guess:
            self.guess = (max(self.y),1,1,0)
        self.perform_fit(guess = self.guess)

        if not self._flag_fit_error:
            if kwargs.get('print_result',False):
                self.print_result()
            if kwargs.get('plot_result',False):
                self.plot_result()
