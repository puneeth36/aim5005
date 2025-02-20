import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # TODO: There is a bug here... Look carefully! 
        return (x - self.minimum) / (self.maximum - self.minimum)
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    
# class StandardScaler:
#     def __init__(self):
#         self.mean = None
#         raise NotImplementedError

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None  

    def fit(self, y):
        """Compute the mean and standard deviation for each feature."""
        y = np.array(y) 
        self.mean = np.mean(y, axis=0) 
        self.std = np.std(y, axis=0, ddof=0)

        self.std = np.where(self.std == 0, 1, self.std)

    def transform(self, y):
        """Standardize the data using the computed mean and std."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet. Call fit(y) first.")
        
        y = np.array(y)  
        return (y - self.mean) / self.std 

    def fit_transform(self, y):
        """Convenience method to fit and transform in one step."""
        self.fit(y)
        return self.transform(y)


import numpy as np

class LabelEncoder:
    def __init__(self):
        self.classes_ = np.array([])  
        self.class_to_index = {} 

    def fit(self, y):
        """Finds unique class labels and assigns them indices."""
        self.classes_ = np.array(sorted(set(y)))
        self.class_to_index = {label: idx for idx, label in enumerate(self.classes_)} 
        return self

    def transform(self, y):
        """Transforms labels into numerical indices."""
        return np.array([self.class_to_index[label] for label in y]) 

    def fit_transform(self, y):
        """Fits the labels and then transforms them."""
        return self.fit(y).transform(y)
