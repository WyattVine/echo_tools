import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

def color_palette():
    # https://refactoring-ui.nyc3.cdn.digitaloceanspaces.com/previews/whats-in-a-color-palette-01.png
    return(['#20639B','#ED553B','#3CAEA3','#F6D55C','#173F5F'])


def update_matplot_style():

    colors = color_palette()
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['axes.prop_cycle'] = cycler('color',colors)
    mpl.rcParams['image.aspect'] = 'auto'
    mpl.rcParams['image.origin'] = 'lower'
    mpl.rcParams['image.cmap'] = 'cycle'
    mpl.rcParams['lines.markersize'] = 10

def generate_axes(shape,**kwargs):
    '''
    Uses subplot2grid to generate multiple axis. Each axis is equal size.
    shape : tuple (num rows, num columns)

    shape (3,2) leads to [ax1,ax2,ax3,ax4,ax5,ax6] where they are arranged like:
    ax1, ax2, ax3
    ax4, ax5, ax6
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
    order = order of polynomial (1, 2 or 3)
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
        '''creates coordinates for plotting'''

        theta = np.linspace(0,2*np.pi,100)
        self.coords = (self.x0 + 1j*self.y0) + self.r*np.exp(1j*theta)