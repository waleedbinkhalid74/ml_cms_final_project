from abc import abstractmethod
from email.errors import ObsoleteHeaderDefect
import numpy as np
import pandas as pd
import scipy
import itertools

class observables_dict:
    def __init__(self, degree) -> None:
        self.degree = degree
    
    @abstractmethod
    def fit(self):
        """Each dictionary of obserables must implement this
        """
        pass
    
    @abstractmethod
    def segregate_observables_from_variable(self):
        pass

class HermitePairs(observables_dict):
    def __init__(self, degree: int) -> None:
        super().__init__(degree)
        self.data = None
        self.observable_data = None
        self.Nk = self.degree**2

    def fit(self, data: pd.DataFrame):
        self.data = data.copy(deep=True)
        self.observable_data = data.copy(deep=True)
        poly_vars = []
        no_vars = len(self.data.columns) - 1
        assert no_vars == 2, 'Method only works for one dependent and one independent variable currently'

        # Calculate the hermite polynimials and store in a list with degree of polynomial
        for vars in range(no_vars):
            poly = []
            for deg in range(self.degree):
                hermite_poly = scipy.special.hermite(deg, monic=False)
                poly.append((deg, hermite_poly(self.data[self.data.columns[vars+1]])))
            poly_vars.append(poly)
        
        hermite_combinations = list(itertools.product(poly_vars[0], poly_vars[1]))
        # Combinations stored in a list of tuple of tuple format [((degree, polynomial_x), (degree, polynomial_y)]
        for index, combination in enumerate(hermite_combinations):
            hermite_prod = combination[0][1]*combination[1][1]
            self.observable_data.loc[:,'H' + str(combination[0][0]) + '(x1)' + 'H' + str(combination[1][0])+ '(x2)'] = hermite_prod
        return self.observable_data
    
    def segregate_observables_from_variable(self, data):
        observables = data.iloc[:,-self.degree**2:]
        observables.insert(0, 'ID', data.iloc[:,0])
        return observables
    
class Identity(observables_dict):
    def __init__(self) -> None:
        super().__init__(1)
        self.data = None
        self.observable_data = None
        self.Nk = 2
        
    def fit(self, data: pd.DataFrame):            
        self.data = data.copy(deep=True)
        self.observable_data = pd.concat([self.data, self.data.iloc[:,1:]], axis=1)
        
        return self.observable_data

    def segregate_observables_from_variable(self, data):
        observables = data.iloc[:,-self.Nk:]
        observables.insert(0, 'ID', data.iloc[:,0])
        return observables
    
class Polynomials(observables_dict):
    def __init__(self, degree) -> None:
        super().__init__(degree)
        self.data = None
        self.observable_data = None
        self.Nk = self.degree**2
        
    def fit(self, data:pd.DataFrame):
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
        return self.observable_data
    
    def segregate_observables_from_variable(self, data):
        observables = data.iloc[:,-self.degree**2:]
        observables.insert(0, 'ID', data.iloc[:,0])
        return observables