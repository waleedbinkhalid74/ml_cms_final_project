from abc import abstractmethod
import numpy as np
import pandas as pd
import scipy
import itertools

class observables_dict:
    def __init__(self, data: pd.DataFrame, degree) -> None:
        self.degree = degree
        self.data = data.copy(deep=True)
        self.observable_data = data.copy(deep=True)
    
    @abstractmethod
    def fit(self):
        """Each dictionary of obserables must implement this
        """
        pass
    

class HermitePairs(observables_dict):
    
    def fit(self):
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