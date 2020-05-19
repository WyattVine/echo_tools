import pandas as pd
import numpy as np

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

        self.Is_raw = self.Is
        self.Qs_raw = self.Qs
        self.time = np.array(self.Is.index)
        self.columns = np.array(self.Is.columns)