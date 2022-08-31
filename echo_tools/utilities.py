import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

def color_palette():
    return(['#20639B','#C4000D','#F68100','#67BC25','#FFC107'])

def update_matplot_style():

    colors = color_palette()
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['axes.prop_cycle'] = cycler('color',colors)
    mpl.rcParams['image.aspect'] = 'auto'
    mpl.rcParams['image.origin'] = 'lower'
    mpl.rcParams['image.cmap'] = 'viridis'
    mpl.rcParams['lines.markersize'] = 3

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

class circle():
    '''
    A circle in the complex plane. Used in plotting functions
    '''

    def __init__(self,r,x0=0,y0=0):
        
        self.r = r
        self.x0 = x0
        self.y0 = y0
        self.create_coords()
        
    def create_coords(self):

        theta = np.linspace(0,2*np.pi,100)
        self.coords = (self.x0 + 1j*self.y0) + self.r*np.exp(1j*theta)