import numpy as np
import pandas as pd
from Observables import observables_dict
import scipy
from tqdm import tqdm
from typing import Tuple
class EDMD:
    def __init__(self, data: pd.DataFrame, dim: int, dict: observables_dict):
        """Initializes the Extended Dynamic Mode class object with training data, dictionary of observables and data dimensions

        Args:
            data (pd.DataFrame): Input data with column structure
            ID      time    Independent_Var_1      ...     Independent_Var_n       Dependent_Var_1     ...     Dependent_Var_n
            
            dim (int): Dimensionality of data
            dict (observables_dict): Dictionary of observables to be applied to the data for the calculation of hte Koopman operator approximation matrix 
        """        
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
            ID      time    Var_1        Var_2      ...     Var_n

        """

        data_x = self.raw_data.iloc[:,:self.dim+2]
        data_y = self.raw_data.iloc[:,self.dim+2:]
        data_y.insert(0, 'ID', data_x.iloc[:,0])        
        data_y.insert(1, 'time', data_x.iloc[:,1])        
        return data_x, data_y

    def construct_G(self) -> np.array:
        """Calculates the G matrix using dictionary of observables applied to the independent variable

        Returns:
            np.array: G matrix
        """
        
        G = np.zeros((self.Nk, self.Nk))
        for id in self.obs_dict_x['ID'].unique():
            obs_m = self.obs_dict_x[self.obs_dict_x['ID'] == id].iloc[:,2:].to_numpy()
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
            obs_m_x = self.obs_dict_x[self.obs_dict_x['ID'] == id].iloc[:,2:].to_numpy()
            obs_m_y = self.obs_dict_y[self.obs_dict_y['ID'] == id].iloc[:,2:].to_numpy()
            A = A + obs_m_x.T @ obs_m_y
        M = self.obs_dict_x['ID'].unique().max() + 1
        A = 1/M * A
        return A
    
    def fit(self, B: np.array=None):
        """Applied the EDMD algorithm as detailed in Williams 2015

        Args:
            B (np.array, optional): Matrix to reconstruct the full state observables from the dictionary of observables. 
            This can also be calculated using least squares if not provided by the user. Defaults to None.
        """

        # Step 1: Construct G and A
        G = self.construct_G()
        A = self.construct_A()
        # Step 2: Calculate K and get eigenvalues and left and right eigenvectors
        K = np.linalg.pinv(G) @ A
        eigenvalues, eigenvectors_right = np.linalg.eig(K)
        # Sort eigenvalues and eigenvectors        
        self.order_eigenvalues_and_vectors(eigenvalues=eigenvalues, eigenvectors=eigenvectors_right)
        # Calculate B if not provided by user
        if B == None:
            self.B = self.estimate_B()
        else:
            self.B = B
        # Step 3: Compute eigenmodes
        self.build_eigenmodes()

    def order_eigenvalues_and_vectors(self, eigenvalues: np.array, eigenvectors: np.array):
        """Orders the eigenvalues and vectors of the Koopman Matrix

        Args:
            eigenvalues (np.array): unsorted eigenvalues
            eigenvectors (np.array): unsorted eigenvectors (right)
        """
        eigvec_list = [eigenvectors[:,i] for i in range(eigenvectors.shape[-1])]
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_idx]
        self.eigenvectors_right = np.array(eigvec_list)[sorted_idx].T
        self.eigenvectors_left_conj = np.linalg.inv(self.eigenvectors_right)

    def build_eigenfunction(self, data: pd.DataFrame) -> np.array:
        """Calculates the all the eigenfunctions stacked as a matrix using $\Psi(x) V$ where $\Psi(x)$ is the matrix from the dictionary of observables
        and V is the matrix containing the eigenvectors.

        Args:
            data (pd.DataFrame): Input data of structure
            ID      time    Var_1        Var_2      ...     Var_n

        Returns:
            np.array: eigenfunctions stacked in a matrix
        """
        eigenfunctions = self.build_dict_from_data(data).to_numpy() @ self.eigenvectors_right
        return eigenfunctions
    
    def calculate_eigenfunction(self, data: pd.DataFrame, eigvec_pos: int) -> np.array:
        """Calculates a specific eigenfunction based on the a specific eigenvector 

        Args:
            data (pd.DataFrame): Input data of structure
            ID      time    Var_1        Var_2      ...     Var_n
            eigvec_pos (int): Position of eigenvector based on the sorted eigenvalues list  

        Returns:
            np.array: single eigenfunction
        """

        eigenfunction = self.build_dict_from_data(data) @ self.eigenvectors_right[:,eigvec_pos]
        return eigenfunction
    
    def build_dict_from_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Dictionary of observables matrix is constructed by applying the observable functions to data

        Args:
            data (pd.DataFrame): Input data of structure
            ID      time    Var_1        Var_2      ...     Var_n
            
        Returns:
            pd.DataFrame: Dictionary of observables in format
            Obs_1        Obs_2      ...     Obs_N_k
        """
        
        dict_obs = self.dict_type.segregate_observables_from_variable(self.dict_type.fit(data)).iloc[:,2:]
        return dict_obs
        
    def predict(self, initial_value_df: pd.DataFrame, t_range: np.array) -> pd.DataFrame:
        """Uses the eigenvalues modes and functions to predict future states of the system given an initial condition

        Args:
            initial_value_df (pd.DataFrame): Initial condition data of structure
            ID      time    Var_1        Var_2      ...     Var_n
            t_end (int): final timestep

        Returns:
            pd.DataFrame: Resulting prediction
        """
        
        final_prediction = pd.DataFrame()#initial_value_df.copy(deep=True)
        for id in tqdm(initial_value_df.ID.unique()):
            prediction_df = initial_value_df[initial_value_df['ID'] == id].copy(deep=True)
            prediction = initial_value_df[initial_value_df['ID'] == id].copy(deep=True)
            mode = self.eigenmodes
            eigenvalues_mat = np.diag(self.eigenvalues)
            for t in t_range[1:]:
                func = self.build_eigenfunction(pd.DataFrame(prediction.iloc[-1,:]).T)
                prediction_mat = func @ eigenvalues_mat @ mode
                prediction = pd.DataFrame(np.insert(prediction_mat ,0, prediction[['ID', 'time']].iloc[0]).real, columns=list(initial_value_df.columns))
                prediction.iloc[-1,1] = t
                prediction_df = pd.concat([prediction_df, prediction])
            final_prediction = pd.concat([final_prediction, prediction_df])
        return final_prediction
        
    def build_eigenmodes(self):
        """Calculates the eigenmodes of the koopman matrix
        """
        eigenmodes = np.matrix(self.eigenvectors_left_conj) @ self.B
        eigenmodes = eigenmodes
        self.eigenmodes = eigenmodes
    
    def estimate_B(self) -> np.array:
        """Estimates the B matrix using least squares

        Returns:
            np.array: B matrix
        """
        B = np.linalg.lstsq(self.obs_dict_x.iloc[:,2:].to_numpy(), self.x_data.iloc[:,2:].to_numpy(), rcond=None)[0]
        return B
    
    def set_B(self, B:np.array):
        """Sets the B matrix if provided by the user.

        Args:
            B (np.array): [description]
        """
        self.B = B
        
    def koopman_matrix(self) -> np.array:
        """Returns the Koopman matrix if required by the user for analysis

        Returns:
            np.array: Koopman matrix
        """
        return self.eigenvectors_right @ np.diag(self.eigenvalues) @ np.linalg.inv(self.eigenvectors_right)