from abc import abstractmethod
import numpy as np
import pandas as pd
import scipy
from scipy import special
import itertools

class observables_dict:
    def __init__(self, degree: int) -> None:
        """Constructor for abstract observables dictionary class

        Args:
            degree (int): Degree of observables. Has different meaning for each dictionary
        """
        self.degree = degree
    
    @abstractmethod
    def fit(self):
        """Each dictionary of obserables must implement this to add observable functions to given data
        """
        pass
    
    @abstractmethod
    def segregate_observables_from_variable(self):
        """Each dictionary of obserables must implement this to remove observable function data from actual data
        """
        pass

class HermitePairs(observables_dict):
    """Forms Hermite Polynomial combinations between the variables

    """
    def __init__(self, degree: int) -> None:
        """Initializes the hermite pair dictionary.

        Args:
            degree (int): number of Hermite polynomials to calculate for each variable
        """
        super().__init__(degree)
        self.data = None
        self.observable_data = None
        self.Nk = self.degree**2

    def fit(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates the first "degree" number of herimte polynomials of both variables and makes comobinations of all the hermite polynomials  

        Args:
            data (pd.DataFrame): dataframe with the column structure as
        ID      Var_1        Var_2

        Returns:
            (pd.DataFrame): Same dataframe as input but with degree**2 additional columns each containing a hermite pair.
        """
        
        self.data = data.copy(deep=True)
        self.observable_data = data.copy(deep=True)
        poly_vars = []
        no_vars = len(self.data.columns) - 1
        assert no_vars == 2, 'Method only works for one dependent and one independent variable currently'

        # Calculate the hermite polynimials and store in a list with degree of polynomial
        for vars in range(no_vars):
            poly = []
            for deg in range(self.degree):
                hermite_poly = special.hermite(deg, monic=False)
                poly.append((deg, hermite_poly(self.data[self.data.columns[vars+1]])))
            poly_vars.append(poly)
        
        hermite_combinations = list(itertools.product(poly_vars[0], poly_vars[1]))
        # Combinations stored in a list of tuple of tuple format [((degree, polynomial_x), (degree, polynomial_y)]
        for index, combination in enumerate(hermite_combinations):
            hermite_prod = combination[0][1]*combination[1][1]
            self.observable_data.loc[:,'H' + str(combination[0][0]) + '(x1)' + 'H' + str(combination[1][0])+ '(x2)'] = hermite_prod
        return self.observable_data
    
    def segregate_observables_from_variable(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove observable function data from actual data

        Args:
            data (pd.DataFrame): dataframe with the column structure as
        ID      Var_1        Var_1        Obs_1       Obs_2       ...       Obs_N_k

        Returns:
            pd.DataFrame: dataframe with the column structure as
        ID        Obs_1       Obs_2       ...       Obs_N_k

        """
        
        observables = data.iloc[:,-self.degree**2:]
        observables.insert(0, 'ID', data.iloc[:,0])
        return observables
    
class Identity(observables_dict):
    """Uses f(x)=x as identity map
    """
    def __init__(self) -> None:
        """Initializes the identity observables
        """
        super().__init__(1)
        self.data = None
        self.observable_data = None
        self.Nk = 2
        
    def fit(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies the identity map

        Args:
            data (pd.DataFrame): dataframe with the column structure as
        ID      Var_1        Var_2      ...     Var_n

        Returns:
            pd.DataFrame: dataframe with the column structure as
        ID      Var_1        Var_2      ...     Var_n       Var_1        Var_2      ...     Var_n

        """
                    
        self.data = data.copy(deep=True)
        self.observable_data = pd.concat([self.data, self.data.iloc[:,1:]], axis=1)
        
        return self.observable_data

    def segregate_observables_from_variable(self, data:pd.DataFrame) -> pd.DataFrame:
        """Remove observable function data from actual data

        Args:
            data (pd.DataFrame): dataframe with the column structure as
        ID      Var_1        Var_2      ...     Var_n       Var_1        Var_2      ...     Var_n

        Returns:
            pd.DataFrame: dataframe with the column structure as
        ID      Var_1        Var_2      ...     Var_n

        """
        
        observables = data.iloc[:,-self.Nk:]
        observables.insert(0, 'ID', data.iloc[:,0])
        return observables
    
class Polynomials(observables_dict):
    """Polynomial dictionary of observables
    """
    def __init__(self, degree: int) -> None:
        """Initializes the polynomial dictionary of observables

        Args:
            degree (int): The highest allowed degree of the constructed polynomials.
        """
        super().__init__(degree)
        self.data = None
        self.observable_data = None
        # TODO: Qais please see if you are able to generalize this for high dimensions
        # Number of observables is the square of the polynomial ONLY if data is two dimensional. In general it should be self.degree**dimensions
        self.Nk = None
        
    def fit(self, data:pd.DataFrame) -> pd.DataFrame:
        """Calculates the first "degree" number of polynomials using all available variables in data  

        Args:
            data (pd.DataFrame): dataframe with the column structure as
        ID      Var_1        Var_2      ...     Var_n

        Returns:
            (pd.DataFrame): Same dataframe as input but with N_k additional columns each containing a hermite pair.
        """

        self.data = data.copy(deep=True)
        self.observable_data = data.copy(deep=True)
        poly_vars = []
        no_vars = len(self.data.columns) - 1
        assert no_vars == 2, 'Method only works for one dependent and one independent variable currently'
        # Calculate the hermite polynimials and store in a list with degree of polynomial
        for vars in range(no_vars):
            poly = []
            for deg in range(0, self.degree+1):
                polynomial = self.data[self.data.columns[vars+1]]**deg
                poly.append((deg, polynomial))
            poly_vars.append(poly)
        poly_combinations = list(itertools.product(poly_vars[0], poly_vars[1]))
        poly_combinations = [comb for comb in poly_combinations if (comb[0][0] + comb[1][0] <= self.degree) and (comb[0][0] + comb[1][0] != 0)]
        # Combinations stored in a list of tuple of tuple format [((degree, polynomial_x), (degree, polynomial_y)]
        for index, combination in enumerate(poly_combinations):
            poly_prod = combination[0][1]*combination[1][1]
            self.observable_data.loc[:,  'x1^' + str(combination[0][0]) + 'x2^' + str(combination[1][0])] = poly_prod
        self.Nk = len(list(self.observable_data.columns)) - 1
        return self.observable_data
    
    def segregate_observables_from_variable(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove observable function data from actual data

        Args:
            data (pd.DataFrame): dataframe with the column structure as
        ID      Var_1        Var_1        Obs_1       Obs_2       ...       Obs_N_k

        Returns:
            pd.DataFrame: dataframe with the column structure as
        ID        Obs_1       Obs_2       ...       Obs_N_k

        """
        
        observables = data.iloc[:,1:]
        observables.insert(0, 'ID', data.iloc[:,0])
        return observables