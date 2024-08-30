import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
from datetime import datetime
import os
import h5py
from .preprocess import Scaler
from .metrics import ssim, psnr
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class vqpca:

    def __init__(self, X, stopping_rule="variance", itmax=200, atol=1e-8, rtol=1e-8, ctol=1e-6):

        # Data matrix
        self.X_ = X
        (nobs, nvars) = np.shape(X)
        self.n_obs_ = nobs
        self.nvars_ = nvars

        # Stopping rule (check existence first)
        stop_list = ["variance", "n_eigs"]
        if stopping_rule not in stop_list:
            raise ValueError("The stopping rule specified does not exist")
        self.stopping_rule_ = stopping_rule

        # Parameters for convergence
        self.itmax_ = itmax
        self.atol_  = atol  
        self.rtol_  = rtol  # Tolerance on reconstruction error
        self.ctol_  = ctol  # Tolerance on centroids


    # Initialization methods
    def initialize_centroids(self, method, Ci=None):

        '''This function will initialize the clusters centroids 
        using an avalibale method. Available methods are: uniform,
         k-means, random '''
                
        if method == 'kmeans':
            # Centroids are initialized using k-means
            kmeans = KMeans(n_clusters=self.k_, random_state=0).fit(self.X_)
            self.labels_ = kmeans.labels_
            self.C_ = kmeans.cluster_centers_

        elif method == 'random':
            np.random.seed(1000)
            # Centroids are initialized randomly
            self.labels_ = np.random.randint(0, self.k_, self.X_.shape[0])
            self.C_ = np.zeros((self.k_, self.X_.shape[1]))
            for i in range(self.k_):
                self.C_[i,:] = np.mean(self.X_[self.labels_==i,:], axis=0)

        elif method == 'uniform':
            # Centroids are initialized uniformly
            npts = self.X_.shape[0]
            # Get number of groups
            ngroups = int(np.floor(npts / self.k_))
            # Get number of points in the last group
            nlast = npts - ngroups * self.k_
            # Initialize labels
            self.labels = np.zeros(npts)
            # Initialize centroids
            self.C_ = np.zeros((self.k_, self.X_.shape[1]))
            # Loop over groups
            for i in range(self.k_):
                # Assign labels
                self.labels[i*ngroups:(i+1)*ngroups] = i
                # Assign centroids
                self.C_[i,:] = np.mean(self.X_[i*ngroups:(i+1)*ngroups,:], axis=0)
            # Assign labels to last group
            self.labels[-nlast:] = self.k_ - 1
            # Assign centroids to last group
            self.C_[-1,:] = np.mean(self.X_[-nlast:,:], axis=0)

        else:
            raise ValueError("Initialization method for centroids not recognized")

        return self
        
    def initialize_pca(self, n_start=2):

        '''This function will initialize the PCA objects in each cluster'''

        pca_list = [] 
        for i in range(self.k_):
            pca = PCA(n_components=n_start)
            idps = np.where(self.labels_==i)[0]
            pca.fit(self.X_[idps,:])
            pca_list.append(pca)

        self.pca_ = pca_list
        return self

    def fit(self, k=2, q=0.99, n_start=2, verbose=True, init_centroids='kmeans'):

        '''The fit method runs the VQPCA routine. Note that the matrix X should be
        previously specified when building the object via constructor
        
        Inputs:
        k (int) = number of clusters
        q (float) = amount of variance or number of principal components

        (Optional)
        n_start (int) = number of pcs used at the first iteration (default=2)
        init_centroids (str) = method for initializing the centroids (default='k-means)
                               available are: 'k-means', 'random', 'uniform'
        verbose (bool) = print information (default=True)

        Outputs:
        labels_new (np array) = vector of integers corresponding to cluters' labels
        '''

        # Number of clusters
        self.k_ = k

        # PCA stopping input
        self.q_ = q

        # Initialize global time
        st_global = time.time()

        # Initialize flag for convergence
        conv = False
        # Initialize iteration counter
        iter = 0

        # Initialize centroids
        self.initialize_centroids(method=init_centroids)
        
        # Initialize pca
        self.initialize_pca(n_start=n_start)

        while conv == False and iter < self.itmax_:

            # Monitor iteration time
            st_iter = time.time()
            # Matrix of projection errors
            eps_matrix = np.zeros((self.X_.shape[0], self.k_))
            # Perform PCA in each cluster and calculate projection error
            for i in range(self.k_):
                # Predict the point using the local PLS model
                U_scores = self.pca_[i].transform(self.X_)
                # Reconstruct X
                X_rec    = self.pca_[i].inverse_transform(U_scores)
                # Calculate norm of the residual
                eps_matrix[:,i] = np.sum((self.X_ - X_rec)**2, axis=1)
            # Assign each point to the cluster with the lowest residual
            labels_new = np.argmin(eps_matrix, axis=1)
            # Calculate the new centroids
            centroids_new = np.zeros((self.k_, self.X_.shape[1]))
            for i in range(self.k_):
                centroids_new[i,:] = np.mean(self.X_[labels_new==i,:], axis=0)
            # Calculate the change in the centroids
            delta_centroids = np.linalg.norm(centroids_new - self.C_) / np.linalg.norm(self.C_)
            if delta_centroids < self.ctol_:
                conv = True
                if verbose:
                    print('Converged because of the centroids')

            # Check convergence on reconstruction error variance
            eps_rec_new = np.min(eps_matrix, axis=1)
            if iter == 0:
                self.eps_ = np.mean(eps_rec_new)
            else:
                delta_eps = np.abs((self.eps_ - np.mean(eps_rec_new)) / self.eps_)
                if delta_eps < self.rtol_:
                    conv = True
                    if verbose:
                        print('Converged because of reconstruction error variance')

            # Update PCA objects
            pca_list = []
            rec_err_clust = []
            for i in range(self.k_):
                pca = PCA(n_components=self.q_)
                pca.fit(self.X_[labels_new==i,:])
                pca_list.append(pca)
                U_local = pca.transform(self.X_[labels_new==i])
                X_local_rec = pca.inverse_transform(U_local)
                rec_err_clust.append(mean_squared_error(self.X_[labels_new==i], X_local_rec))
                
            # Update self
            self.pca_ = pca_list
            self.C_ = centroids_new
            self.labels_ = labels_new
            self.eps_    = np.mean(eps_rec_new)
            # Get components
            self.A_ = self.get_components()

            # Update iteration counter 
            iter += 1
            if iter == self.itmax_:
                print("Iterations reached maximum allowed number")

            # Time for iteration
            et_iter = time.time()
            dt_iter = et_iter - st_iter

            if verbose == True:
                print("Iteration: ", iter)
                print("Mean squared global reconstruction error = ", self.eps_ )
                print("Mean squared error in each cluster: ")
                print(rec_err_clust)
                if iter > 1:
                    print("Reconstruction error variance = ", delta_eps)
                    print("Centroids variance = ", delta_centroids)
                print(f"Elapsed time for iteration: {dt_iter} seconds \n \n")

        # Measure time at convergence
        et_global = time.time()
        dt_global = et_global - st_global
        if verbose == True:
            print(f"Elapsed time for global convergence: {dt_global} seconds")

        # Display information
        if verbose:
            print("Obtained reconstruction error = ", self.eps_)

        return labels_new, self
    
    def reconstruct(self, X=None):

        '''This function will provide the reconstructed data. If the X is not
        given, then the reconstructed data will be the original ones
        
        Inputs:
        X (2D NumPy array) = data matrix, can also be a new one. If None, the self.X_ matrix is used (default=None)'''

        # Check if X was given
        if X.all() == None:
            X = self.X_

        X_rec = np.zeros_like(X)
        for i in range(self.k_):
            U_local = self.pca_[i].transform(X[self.labels_==i])
            X_rec[self.labels_==i] = self.pca_[i].inverse_transform(U_local)

        return X_rec
    
    def predict(self, X):

        '''This function will cluster a new matrix X based
         on the local basis that the local PCA found '''
        
        # Check shape of X
        (nobs, nvars) = np.shape(X)
        if nvars != self.nvars_:
            raise ValueError("Number of columns of new matrix must be the same as original")
        
        # Matrix of projection errors
        eps_matrix = np.zeros((X.shape[0], self.k_))
        # Perform PCA in each cluster and calculate projection error
        for i in range(self.k_):
            # Predict the point using the local PLS model
            U_scores = self.pca_[i].transform(X)
            # Reconstruct X
            X_rec    = self.pca_[i].inverse_transform(U_scores)
            # Calculate norm of the residual
            eps_matrix[:,i] = np.sum((X - X_rec)**2, axis=1)

        # Assign each point to the cluster with the lowest residual
        labels = np.argmin(eps_matrix, axis=1)

        return labels

    ### Function for getting access to attributes ###
    def get_components(self):

        '''This function will return the PCA components'''

        components = []
        for i in range(self.k_):
            ci = self.pca_[i].components_.T
            components.append(ci)
        
        return components
    
    def write_output_files(self, foldername=None, format='h5'):

        ''' This function will create some output files. Available formats are
            h5 and txt. h5 is recommended as it uses less memory, however might be
            trickier to import. If your dataset is small, you can use .txt without
            much problems '''
        
        if foldername == None:
            # Get the current date
            current_date = datetime.now().strftime("%Y-%m-%d")
            foldername = "VQPCA_k" + str(self.k_) + "q_" + str(self.q_) + '_' + current_date
        
        # Create directory if it does not exists
        if os.path.exists(foldername) == False:
            os.mkdir(foldername)

        if format == "h5":
            file_path = foldername + '/output.h5'
            with h5py.File(file_path, 'w') as f:

                # Write components
                components = self.get_components()
                basis_group = f.create_group("components")
                for i, array in enumerate(components):
                    basis_group.create_dataset(f'array_{i}', data=array)

                # Write labels
                f.create_dataset("labels", data=self.labels_)

                # Reconstruction error
                f.create_dataset("reconstruction_error", data=self.eps_)

                # Write centroids
                f.create_dataset("centroids", data=self.C_)

        elif format == "txt":
            
            np.savetxt(foldername + '/labels.txt', self.labels_, fmt='%d')
            np.savetxt(foldername + '/centroids.txt', self.C_)

        return self
    
class vqpls:

    '''This is the class that implements the Vector Quantization Partial Least Square routine.'''

    # Default constructor
    def __init__(self, ctol=1e-6, Rtol=1e-6,
                  itmax=100, verbose=True, early_stopping=False, patience=10):
        
        self.ctol = ctol            # Centroids tolerance for convergence
        self.Rtol = Rtol            # R2 tolerance for convergence
        self.itmax = itmax          # maximum number of iteration
        self.verbose = verbose      # Verbosity level

        # Monitor for early stopping
        self.es = early_stopping    # Early stopping flag
        self.patience = patience    # Number of iteration for early stopping


    def initialize_centroids(self, X, init='kmeans', Y=None):

        ''' Function to initialize the clusters. Available 
        options are:
        - kmeans: initialize with kmeans
        - random: pick k random centroids
        - uniform: pick k centroids sampling uniformly
        - global_pls: perform global PLS and apply k-means on the PLS scores'''

        if init == 'kmeans':
            # Centroids are initialized using k-means
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(X)
            self.labels = kmeans.labels_
            self.centroids = kmeans.cluster_centers_

        elif init == 'random':
            # Centroids are initialized randomly
            self.labels = np.random.randint(0, self.n_clusters, X.shape[0])
            self.centroids = np.zeros((self.n_clusters, X.shape[1]))
            for i in range(self.n_clusters):
                self.centroids[i,:] = np.mean(X[self.labels==i,:], axis=0)

        elif init == 'uniform':
            # Centroids are initialized uniformly
            npts = X.shape[0]
            # Get number of groups
            ngroups = int(np.floor(npts / self.n_clusters))
            # Get number of points in the last group
            nlast = npts - ngroups * self.n_clusters
            # Initialize labels
            self.labels = np.zeros(npts)
            # Initialize centroids
            self.centroids = np.zeros((self.n_clusters, X.shape[1]))
            # Loop over groups
            for i in range(self.n_clusters):
                # Assign labels
                self.labels[i*ngroups:(i+1)*ngroups] = i
                # Assign centroids
                self.centroids[i,:] = np.mean(X[i*ngroups:(i+1)*ngroups,:], axis=0)
            # Assign labels to last group
            self.labels[-nlast:] = self.n_clusters - 1
            # Assign centroids to last group
            self.centroids[-1,:] = np.mean(X[-nlast:,:], axis=0)

        elif init == 'global_pls':

            # Fit a global PLS model
            pls = PLSRegression(n_components=self.n_components, scale=False, tol=self.Rtol)
            # Check if Y is provided
            if Y is None:
                raise ValueError('Y must be provided when using global PLS')
            
            # Fit the model
            pls.fit(X, Y)

            # Get the scores
            T = pls.x_scores_

            # Perform kmeans on the scores
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(T)

            # Assign labels
            self.labels = kmeans.labels_

            # Initialize centroids
            self.centroids = np.zeros((self.n_clusters, X.shape[1]))
            for i in range(self.n_clusters):
                self.centroids[i,:] = np.mean(X[self.labels==i,:], axis=0)

        else:
            raise ValueError('Initialization method not recognized')

        return self
    
    def initialize_pls(self, X, Y):
        '''This function initializes the PLS bases in the clusters'''

        self.pls = []
        for i in range(self.n_clusters):
            self.pls.append(PLSRegression(n_components=self.n_components, scale=False, tol=self.Rtol))
            idps = np.where(self.labels==i)[0]
            self.pls[i].fit(X[idps,:], Y[idps,:])

        return self
    
    def fit(self, X, Y, init='kmeans', n_clusters=2, n_components=3):

        ''' This function performs the VQPLS algorithm, which
        means assigning each point to a cluster and then fitting
        a PLS model to each cluster, such that the sum of the
        squared residuals is minimized.
        
        Inputs:
        X (2D NumPy array): independent variables data matrix
        Y (2D NumPy array): target variables data matrix
        init (str): initialization method (available are 'kmeans', 'random', 'global_pls') (default='kmeanse')
        n_clusters (int): number of clusters (default=2)
        n_components (int): number of retained components (default=3)
        
        Outputs:
        self: updated class with labels and PLS models'''

        # Initialize class attributes
        self.init = init
        self.n_clusters = n_clusters
        self.n_components = n_components

        # Check shapes of X and Y
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same number of rows')
        
        self.nX = X.shape[1]
        self.nY = Y.shape[1]

        # Initialize time
        tic = time.time()

        # Initialize centroids
        if self.init == 'global_pls':
            self.initialize_centroids(X, self.init, Y)
        else:
            self.initialize_centroids(X, self.init)

        # Initialize PLS models
        self.initialize_pls(X, Y)

        # Initialize convergence flag and iteration counter
        converged = False
        it = 0

        # Save history
        hs = []
        if self.es == True:
            n_es = 0            # Number of times over the minimum detected

        # Loop until convergence
        while converged == False and it < self.itmax:

            R_matrix = np.zeros((X.shape[0], self.n_clusters))

            for i in range(self.n_clusters):
                    
                # Predict the point using the local PLS model
                Y_pred = self.pls[i].predict(X)

                # Calculate norm of the residual
                R_matrix[:,i] = np.sum((Y - Y_pred)**2, axis=1)
        
            # Assign each point to the cluster with the lowest residual
            labels_new = np.argmin(R_matrix, axis=1)
            
            # Calculate the new centroids
            centroids_new = np.zeros((self.n_clusters, X.shape[1]))
            for i in range(self.n_clusters):
                centroids_new[i,:] = np.mean(X[labels_new==i,:], axis=0)
            
            # Calculate the change in the centroids
            delta_centroids = np.linalg.norm(centroids_new - self.centroids) / np.linalg.norm(self.centroids)
            if delta_centroids < self.ctol:
                converged = True
                print('Converged because of the centroids')

            ### Get the score 
            if it == 0:
                self.score(X, Y)
                mse = self.mse
                r2  = self.r2
                hs.append(mse)
            elif it > 0:
                r2_old  = self.r2
                self.score(X, Y)
                r2_new = self.r2
                mse_new = self.mse
                mse_tol = 1e-6
                delta_mse = np.abs((mse - mse_new))/mse
                delta_r2  = np.abs((r2_old - r2_new))/r2_old
                if delta_r2 == 0:
                    raise ValueError("Smething is wrong with convergence")
                if delta_mse < mse_tol or delta_r2 < self.Rtol:
                    converged = True
                    print('Converged since MSE did not change')

                hs.append(mse_new)

            ### Check convergence for early stopping
            if self.es == True and it > 0:
                delta = hs[it] - hs[it-1]
                if delta > 0:
                    n_es += 1
                if n_es > self.patience:
                    converged = True
                    print('Algorithm converged for early stopping')

            ### Convergence for max iterations
            if it == self.itmax:
                print('Maximum number of iterations reached')

            else:
                self.centroids = centroids_new
                self.labels = labels_new
            
            # Update the PLS model
            for i in range(self.n_clusters):
                self.pls[i].fit(X[self.labels==i,:], Y[self.labels==i,:])
            
            # Update the iteration counter
            it += 1

            if self.verbose == True:
                print('Iteration: ', it)
                print('Delta centroids: ', delta_centroids)
                print('Mean squared error: ', self.mse)
                print('')

            
        toc = time.time()
        print('Elapsed time: ', toc-tic)

        return self
    
    def fit_from_labels(self, X, Y, labels, n_components=3):

        ''' This function train the local PLS models
        given another set of labels, without performing the
        VQPLS routine (might be useful when loading previously
        saved idx solutions on large data)
        
        Inputs:
        X (2D NumPy array): independent variables data matrix
        Y (2D NumPy array): target variables data matrix
        labels (1D NumPy array): cluster labels vector
        n_components (int): number of retained components (default=3)
        
        Output:
        self (vqpls class): updated class'''

        # Check shapes of X and Y
        if X.shape[0] != Y.shape[0]:
            raise ValueError('X and Y must have the same number of rows')
        
        self.nX = X.shape[1]
        self.nY = Y.shape[1]

        # Change number of clusters
        if self.n_clusters != max(labels)+1:
            # Update number of clusters
            self.n_clusters = max(labels)+1
            raise Warning("VQPLS is set up with a different number of clusters than the current labels vector")
        # Change labels
        self.labels = labels
        # Calculate PLS models
        self.pls = []
        for i in range(self.n_clusters):
            self.pls.append(PLSRegression(n_components=n_components, scale=False, tol=self.Rtol))
            idps = np.where(self.labels==i)[0]
            self.pls[i].fit(X[idps,:], Y[idps,:])
        # Calculate centroids
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            centroids[i,:] = np.mean(X[labels==i,:], axis=0)
        self.centroids = centroids

        return self

    def predict(self, X):

        '''This function predicts Y (in PC transport, the source term) given
        a matrix of data X'''

        # Initialize Y_pred
        Y_pred = np.zeros((X.shape[0], self.nY))

        # Loop over clusters
        for i in range(self.n_clusters):
            # Get indices of points in the cluster
            idps = np.where(self.labels==i)[0]
            # Predict the points
            Y_pred[idps,:] = self.pls[i].predict(X[idps,:])

        return Y_pred
    
    def score(self, X, Y):

        '''This function returns the different scores of the model globally, namely the R2 score, 
        the mean squared error and the mean absolute error'''

        # Initialize Y_pred
        Y_pred = self.predict(X)

        # Calculate the scores
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        self.r2 = r2_score(Y, Y_pred)
        self.mse = mean_squared_error(Y, Y_pred)
        self.mae = mean_absolute_error(Y, Y_pred)

        print('R2: ', self.r2)
        print('MSE: ', self.mse)
        print('MAE: ', self.mae)

        return self
    
    def score_cluster(self, X, Y):

        '''This function returns the model scores in the clusters'''

        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

        # Initialize Y_pred
        Y_pred = self.predict(X)

        # Initialize the metrics in the clusters as empty lists
        r2_clust  = []
        mse_clust = []
        mae_clust = []

        # Gt labels
        for i in range(self.n_clusters):
            r2_clust.append(r2_score(Y[self.labels==i,:], Y_pred[self.labels==i,:]))
            mse_clust.append(mean_squared_error(Y[self.labels==i,:], Y_pred[self.labels==i,:]))
            mae_clust.append(mean_absolute_error(Y[self.labels==i,:], Y_pred[self.labels==i,:]))

        return r2_clust, mse_clust, mae_clust
    
    def transform(self, X):

        '''This function projects the data X onto the X score space (Z scores)'''

        # Initialize Z_score
        Z_score = np.zeros((X.shape[0], self.n_components))

        # Loop over clusters
        for i in range(self.n_clusters):
            # Get indices of points in the cluster
            idps = np.where(self.labels==i)[0]
            # Predict the points
            Z_score[idps,:] = self.pls[i].transform(X[idps,:])
        
        return Z_score
    
    ###### Linear models section ######
    def train_linear_models(self, X, Y, test_size=0.20, regressor='linear', random_state=0):
        
        '''This function  will train a linear regression model in each 
        cluster. Available regressors are:
        linear regression, Lasso  regression, Ridge regression'''

        from sklearn.linear_model import LinearRegression, Lasso, Ridge
        # Check if labels exists already as attribute
        if hasattr(self, 'labels') == False:
            raise ValueError("VQPLS model needs to be trained first")
        # Check dimensions
        nobsX, nX = np.shape(X)
        nobsY, nY = np.shape(Y)
        if nobsX != nobsY:
            raise ValueError("Check dimensions of X and Y. Their number of rows must agree")
        if nobsX != len(self.labels):
            raise ValueError('For training, the X must be the original VQPLS X matrix')
        # Train local models
        local_linear_models = []
        local_mse = []
        local_r2  = []
        for i in range(self.n_clusters):
            # Get local data to perform regression
            X_reg = X[self.labels==i,:]; Y_reg = Y[self.labels==i,:]
            # Split training and testing
            X_train, X_test, Y_train, Y_test = train_test_split(X_reg, Y_reg, test_size=test_size, random_state=random_state)
            # Initialize the model
            if regressor == 'linear':
                model = LinearRegression()
            elif regressor == 'Lasso':
                model = Lasso()
            elif regressor == 'Ridge':
                model = Ridge()
            else:
                print('Regressor not recognized. Linear regression will be used')
                model = LinearRegression()
            # Fit the model on the training dataset
            model.fit(X_train, Y_train)
            # Predict test set
            Y_pred = model.predict(X_test)
            # Calculate MSE
            mse = mean_squared_error(Y_test, Y_pred)
            # Calculate R2
            r2  = r2_score(Y_test, Y_pred)
            # Append 
            local_linear_models.append(model)
            local_mse.append(mse)
            local_r2.append(r2)
        # Return models and scores
        self.linear_models = local_linear_models
        return local_linear_models, local_mse, local_r2
    
    def predict_linear_models(self, X, Y):
        '''Use this only on the training data'''
        # Check if labels and linear models are already an attribute
        if hasattr(self, 'labels') == False:
            raise ValueError("Train the VQPLS model first using fit(X,Y)")
        if hasattr(self, 'linear_models') == False:
            print("Local Linear Models not avaialble. Training the models...")
            self.train_linear_models(X, Y)
            print("Done.")
        # Initialize predicted Y
        Y_pred = np.zeros_like(Y)
        for i in range(self.n_clusters):
            Y_pred[self.labels==i,:] = self.linear_models[i].predict(X[self.labels==i,:])
        
        return Y_pred
    
    def predict_linear_models_unseen(self, X):
        ''' This function  predicts new points by assigning
        them using the previously trained classifier'''
        # Check if labels and linear models are already an attribute
        if hasattr(self, 'labels') == False:
            raise ValueError("Train the VQPLS model first using fit(X,Y)")
        if hasattr(self, 'linear_models') == False:
            raise ValueError("Train the local linear models first using train_linear_models(X,Y)")
        if hasattr(self, 'classifier') == False:
            raise ValueError('Train the classifier first using train_classifier(X)')
        
        # Classify observations
        idxc = self.classify_unseen(X)
        # Initialize prediction matrix
        Y_pred = np.zeros((len(X), self.nY))
        for i in range(len(idxc)):
            Y_pred[i,:] = self.linear_models[idxc[i]].predict(X[i,:].reshape((1,-1)))
        return Y_pred

    ##### Classification section #####
    def train_classifier(self, X, test_size=0.20, random_state=0,
                         n_estimators=100):
        
        '''This function will train a random forest classifier
        on the training data used for the VQPLS routine. The classifier will
        be used to assign unseen observations to clusters and predict them
        using the correct model
        
        Inputs:
        X (NumPy array): data matrix
        test_size (float): test data relative size (default=0.2)
        random_state (int): random seed (default=0)
        n_estimators (int): number of estimators in the decision tree (default=100)
        
        Output:
        rf (RandomForestClassifier): the RFC in the scikit-learn implementation'''

        # Check if labels and linear models are already an attribute
        if hasattr(self, 'labels') == False:
            raise ValueError("Train the VQPLS model first using fit(X,Y)")
        
        from sklearn.ensemble import RandomForestClassifier
        # Initialize the object
        rf = RandomForestClassifier(n_estimators=n_estimators)
        # Split train and test
        X_train, X_test, y_train, y_test = train_test_split(X, self.labels, 
                                                test_size=test_size, random_state=random_state)
        # Fit
        print('Training the classifier...')
        tic = time.time()
        rf.fit(X_train, y_train)
        toc = time.time()
        execTime = toc-tic
        print('Classifier trained in ', execTime, ' s')
        # Check score
        print(rf.score(X_test, y_test))
        # Save object
        self.classifier = rf
        return rf, self
    
    ### Classification of unseen observations ###
    def classify_unseen(self, X):
        # Check if classifier is istance
        if hasattr(self, 'classifier') == False:
            raise ValueError("A classifier needs to be trained first using the train_classifier(X) function")

        idxc = self.classifier.predict(X)
        return idxc
    


    

            

        
        



    

        





        




                







            



    

    

        












    

            
        







        

        




        

        

        

        



        



