import numpy as np
import pandas as pd
from Observables import observables_dict
import scipy
from typing import Tuple

class EDMD:
    def __init__(self, data: pd.DataFrame, dim: int, dict: observables_dict):
        """Initializes the Extended Dynamic Mode class object with training data, dictionary of observables and data dimensions

        Args:
            data (pd.DataFrame): Input data with column structure
            ID      Independent_Var_1      ...     Independent_Var_n       Dependent_Var_1     ...     Dependent_Var_n
            
            dim (int): Dimensionality of data
            dict (observables_dict): Dictionary of observables to be applied to the data for the calculation of hte Koopman operator approximation matrix 
        """
        # TODO: Data dimensions should be infered and should not have to be provided by user. Moreover the algorithm should be applied to data of all dimensionality
        # TODO: Qais please see if you can remove this.
        
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
        self.eigenvectors_left_conj = None

    def segregate_xy(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Breaks one dataframe into two, first containing the independent variable and the second containing the dependent variable

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The independent and dependent variable in separate dataframes. both dataframes have ID as their first column and have a column structure as 
            ID      Var_1        Var_2      ...     Var_n

        """

        data_x = self.raw_data.iloc[:,0:self.dim+1]
        data_y = self.raw_data.iloc[:,self.dim+1:]
        data_y.insert(0, 'ID', data_x.iloc[:,0])        
        return data_x, data_y

    def construct_G(self) -> np.array:
        """Calculates the G matrix using dictionary of observables applied to the independent variable

        Returns:
            np.array: G matrix
        """
        
        G = np.zeros((self.Nk, self.Nk))
        for id in self.obs_dict_x['ID'].unique():
            obs_m = self.obs_dict_x[self.obs_dict_x['ID'] == id].iloc[:,1:].to_numpy()
            G = G + obs_m.T @ obs_m
        M = self.obs_dict_x['ID'].unique().max() + 1
        G = 1/M * G
        return G

    def construct_A(self) -> np.array:
        """Calculates the A matrix using dictionary of observables applied to the independent and dependent variable

        Returns:
            np.array: A matrix
        """
        
        A = np.zeros((self.Nk, self.Nk))
        for id in self.obs_dict_x['ID'].unique():
            obs_m_x = self.obs_dict_x[self.obs_dict_x['ID'] == id].iloc[:,1:].to_numpy()
            obs_m_y = self.obs_dict_y[self.obs_dict_y['ID'] == id].iloc[:,1:].to_numpy()
            A = A + obs_m_x.T @ obs_m_y
        M = self.obs_dict_x['ID'].unique().max() + 1
        A = 1/M * A
        return A
    
    def fit(self, B=None):
        # Step 1: Construct G and A
        G = self.construct_G()
        A = self.construct_A()
        # Step 2: Calculate K and get eigenvalues and left and right eigenvectors
        K = np.linalg.pinv(G) @ A
        eigenvalues, eigenvectors_right = scipy.linalg.eig(K)
        # Sort eigenvalues and eigenvectors        
        eigvec_right_list = [eigenvectors_right[:,i] for i in range(eigenvectors_right.shape[-1])]
        # eigvec_left_list = [eigenvectors_left[:,i] for i in range(eigenvectors_left.shape[-1])]
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_idx]
        self.eigenvectors_right = np.array(eigvec_right_list)[sorted_idx].T
        self.eigenvectors_left_conj = np.linalg.inv(self.eigenvectors_right)
        
        if B == None:
            self.B = self.estimate_B()
        else:
            self.B = B
        # Step 3: Compute eigenmodes

    def order_eigenvalues_and_vectors(self, eigenvalues, eigenvectors):
        pass
        
    def build_eigenfunction(self, data: pd.DataFrame):        
        eigenfunctions = self.build_dict_from_data(data).to_numpy() @ self.eigenvectors_right
        return eigenfunctions
    
    def calculate_eigenfunction(self, data: pd.DataFrame, eigvec_pos: int):
        eigenfunction = self.build_dict_from_data(data) @ self.eigenvectors_right[:,eigvec_pos]
        return eigenfunction
    
    def build_dict_from_data(self, data):
        dict_obs = self.dict_type.segregate_observables_from_variable(self.dict_type.fit(data)).iloc[:,1:]
        return dict_obs
        
    def predict(self, initial_value_df, t_end: int):
        
        final_prediction = initial_value_df.copy(deep=True)
        # print(len(final_prediction))
        for id in initial_value_df.ID.unique():
            prediction_df = initial_value_df[initial_value_df['ID'] == id].copy(deep=True)
            prediction = initial_value_df[initial_value_df['ID'] == id].copy(deep=True)
            mode = self.build_eigenmodes()
            eigenvalues_mat = np.diag(self.eigenvalues)
            for i in range(t_end):
                func = self.build_eigenfunction(pd.DataFrame(prediction.iloc[-1,:]).T)
                prediction_mat = func @ eigenvalues_mat @ mode            
                prediction = pd.DataFrame(np.insert(prediction_mat ,0, prediction['ID'].iloc[0]).real, columns=list(initial_value_df))
                prediction_df = pd.concat([prediction_df, prediction])
            final_prediction = pd.concat([final_prediction, prediction_df])
            # print(len(final_prediction))
        # return prediction_df
        return final_prediction
        
    def build_eigenmodes(self):
        eigenmodes = np.matrix(self.eigenvectors_left_conj) @ self.B
        eigenmodes = eigenmodes #/ np.linalg.norm(eigenmodes)
        return eigenmodes
        # return np.matrix(np.linalg.inv(self.eigenvectors_right))
    
    def estimate_B(self):
        B = np.linalg.lstsq(self.obs_dict_x.iloc[:,1:].to_numpy(), self.x_data.iloc[:,1:].to_numpy(), rcond=None)[0]
        return B
    
    def set_B(self, B:np.array):
        self.B = B
        
    def koopman_matrix(self):
        return self.eigenvectors_right @ np.diag(self.eigenvalues) @ np.linalg.inv(self.eigenvectors_right)