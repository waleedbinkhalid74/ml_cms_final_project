import numpy as np
import pandas as pd
from Observables import observables_dict
import scipy


class EDMD:
    def __init__(self, data: pd.DataFrame, dim: int, dict: observables_dict):
        self.raw_data = data # Actual data as Dataframe
        self.dim = dim # Dimensionality of data
        self.x_data, self.y_data = self.segregate_xy()
        self.dict_type = dict
        self.observables_x = self.dict_type.fit(self.x_data)
        self.observables_y = self.dict_type.fit(self.y_data)
        self.obs_dict_x = self.dict_type.segregate_observables_from_variable(self.observables_x)
        self.obs_dict_y = self.dict_type.segregate_observables_from_variable(self.observables_y)
        self.Nk = self.dict_type.Nk
        self.eigenvalues = None
        self.eigenvectors_right = None
        self.eigenvectors_left = None

    def segregate_xy(self):
        data_x = self.raw_data.iloc[:,0:self.dim+1]
        data_y = self.raw_data.iloc[:,self.dim+1:]
        data_y.insert(0, 'ID', data_x.iloc[:,0])        
        return data_x, data_y

    def construct_G(self):
        G = np.zeros((self.Nk, self.Nk))
        for id in self.obs_dict_x['ID'].unique():
            obs_m = self.obs_dict_x[self.obs_dict_x['ID'] == id].iloc[:,1:].to_numpy()
            G = G + obs_m.T @ obs_m
        M = self.obs_dict_x['ID'].unique().max() + 1
        G = 1/M * G
        return G

    def construct_A(self):
        A = np.zeros((self.Nk, self.Nk))
        for id in self.obs_dict_x['ID'].unique():
            obs_m_x = self.obs_dict_x[self.obs_dict_x['ID'] == id].iloc[:,1:].to_numpy()
            obs_m_y = self.obs_dict_y[self.obs_dict_y['ID'] == id].iloc[:,1:].to_numpy()
            A = A + obs_m_x.T @ obs_m_y
        M = self.obs_dict_x['ID'].unique().max() + 1
        A = 1/M * A
        return A
    
    def fit(self):
        # Step 1: Construct G and A
        G = self.construct_G()
        A = self.construct_A()
        # Step 2: Calculate K and get eigenvalues and left and right eigenvectors
        K = np.linalg.pinv(G) @ A
        # self.eigenvalues, self.eigenvectors_left, self.eigenvectors_right = scipy.linalg.eig(K, left=True)
        eigenvalues, eigenvectors_left, eigenvectors_right = scipy.linalg.eig(K, left=True)
        
        # Sort eigenvalues and eigenvectors        
        eigvec_right_list = [eigenvectors_right[:,i] for i in range(eigenvectors_right.shape[-1])]
        eigvec_left_list = [eigenvectors_left[:,i] for i in range(eigenvectors_left.shape[-1])]
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_idx]
        self.eigenvectors_right = np.array(eigvec_right_list)[sorted_idx]
        self.eigenvectors_left = np.array(eigvec_left_list)[sorted_idx]
        # Step 3: Compute eigenmodes

    def build_eigenfunction(self, data: pd.DataFrame, eigvec_loc: int):
        eigenfunction = self.dict_type.segregate_observables_from_variable(self.dict_type.fit(data)).iloc[:,1:].to_numpy() @ self.eigenvectors_right[eigvec_loc, :]
        return eigenfunction