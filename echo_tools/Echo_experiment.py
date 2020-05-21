import pandas as pd
import numpy as np

class Echo_experiment():

    '''
    Base class for experiments
    '''

    def __init__(self,**kwargs):
        self.data_loc = kwargs.get('data_loc',None)
        self.save_loc = kwargs.get('save_loc',None)
        self.data_name_convention = kwargs.get('data_name_convention', 'Is')
        self.data_file_type = kwargs.get('data_file_type', 'pkl')
        self.noise_range = kwargs.get('noise_range',None)

        if self.data_loc:
            self.read_data()

    def read_data(self,**kwargs):

        if self.data_file_type == 'pkl':
            self.Is = pd.read_pickle(self.data_loc + self.data_name_convention + '.pkl')
            self.Qs = pd.read_pickle(self.data_loc + self.data_name_convention.replace('I','Q') + '.pkl')
        elif self.data_file_type == 'csv':
            self.Is = pd.read_csv(self.data_loc + self.data_name_convention + '.csv',index_col=0)
            self.Qs = pd.read_csv(self.data_loc + self.data_name_convention.replace('I','Q') + '.csv',index_col=0)

        self.time = np.array(self.Is.index)
        self.columns = np.array(self.Is.columns)

    def generate_IQs(self):
        return((self.Is ** 2 + self.Qs ** 2).apply(np.sqrt))

    def trim(self,t1,t2):
        '''
        Trims self.Is and self.Qs to only include times between t1 and t2 (e.g. to cut out ringdown)
        and cuts columns with NaN values (indicating a measurement didn't finish)
        '''

        self.Is = self.Is.loc[t1:t2,:].dropna(axis=1)
        self.Qs = self.Qs.loc[t1:t2, :].dropna(axis=1)
        self.time = np.array(self.Is.index)
        self.columns = np.array(self.Is.columns)

    def relabel_columns(self,new_columns):
        '''
        Relabel the columns of self.Is and self.Qs
        '''

        if len(self.columns) != len(new_columns):
            raise ValueError('Number of new column names provided is {}, '
                             'number of column names required is {}'.format(len(new_columns),len(self.columns)))

        _map = {self.columns[i]:new_columns[i] for i in range(len(self.columns))}
        self.Is = self.Is.rename(columns=_map)
        self.Qs = self.Qs.rename(columns=_map)
        self.columns = new_columns

    def relabel_indices(self,new_indices):
        '''
        Relabel the indices of self.Is and self.Qs
        '''

        if len(self.time) != len(new_indices):
            raise ValueError('Number of new index names provided is {}, '
                             'number of index names required is {}'.format(len(new_indices), len(self.time)))

        _map = {self.index[i]:new_indices[i] for i in range(len(self.index))}
        self.Is = self.Is.rename(index=_map)
        self.Qs = self.Qs.rename(index=_map)
        self.time = new_indices
