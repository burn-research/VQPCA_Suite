import numpy as np

class Scaler:
    '''This class can be used to pre-process data by performing centering and scaling
    with different methods available'''

    # Default constructor
    def __init__(self, method, center=True):
        self.method = method
        self.center = center

    # Fit function
    def fit(self, X):
        # Number of columns
        ncols = np.shape(X)[1]
        # Check if center is true
        if self.center == True:
            c = np.mean(X,0)
        else:
            c = np.zeros(ncols)

        # Calculate the scaling value according to X
        if self.method == 'auto':
            gamma = np.std(X,0)
        elif self.method == 'pareto':
            gamma = np.sqrt(np.std(X,0))
        elif self.method == 'range':
            gamma = np.max(X,0)-np.min(X,0)
        elif self.method == 'vast':
            gamma = np.std(X,0)*np.mean(X,0)
        elif self.method == 'max':
            gamma = np.max(X,0)
        else:
            print('No method specified or the method was not recognized')
            gamma = np.ones(ncols)
        
        # Check if any of gamma is lower than 1e-16
        for i in range(len(gamma)):
            if abs(gamma[i]) < 1e-16:
                gamma[i] = 1e-16
                Warning('Division by zero avoided')

        # Assign c to the class
        self.c = c
        self.gamma = gamma

        return self
    
    # Fit_transform function
    def fit_transform(self, X, center=True):
        # Fit
        self.fit(X)
        # Transform
        if center == True:
            Xs = (X - self.c) / self.gamma
        else:
            Xs = X / self.gamma
        return Xs
    
    # Transform function
    def transform(self, X, center=True):
        # Check if c and gamma exists already
        if hasattr(self, 'gamma') and hasattr(self, 'c') == False:
            raise ValueError("Fit the scaler before transforming other data using fit()")
        else:
            if center == True:
                Xs = (X - self.c) / self.gamma
            else:   
                Xs = X / self.gamma
        return Xs
    
    # Inverse transformation
    def inverse_transform(self, X):

        # Check if c and gamma exists already
        if hasattr(self, 'gamma') and hasattr(self, 'c') == False:
            raise ValueError("Fit the scaler before transforming other data using fit()")
        Xi = (X) * self.gamma + self.c
        return Xi
    


    

    







        
            
        


            



        

    

