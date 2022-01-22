import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy
import itertools


class EDMD:
    def __init__(self, data: pd.DataFrame, dim):
        self.raw_data = data # Actual data as Dataframe
        self.dim = dim # Dimensionality of data
        self.x_data, self.x_data = self.segregate_xy(self)
        
    def segregate_xy(self):
        data_x = self.raw_data.iloc[:,0:self.dim+1]
        data_y = self.raw_data.iloc[:,self.dim+1:]
        data_y.insert(0, 'ID', data_x.iloc[:,0])        
        return data_x, data_y
    
    # def apply_dict():
    
    def make_combinations(list1, list2):
        all_combinations = []
        list1_permutations = itertools.permutations(list1, len(list2))
        for each_permutation in list1_permutations:
            zipped = zip(each_permutation, list2)
            all_combinations.append(list(zipped))

        list_combinations = [item for sublist in all_combinations for item in sublist]
        return list_combinations
        