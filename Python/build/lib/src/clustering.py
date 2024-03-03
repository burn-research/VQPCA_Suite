import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from datetime import datetime

class vqpca:

    def __init__(self, X, k, stopping_rule="variance", q=0.99, itmax=200, atol=1e-8, rtol=1e-8, ctol=1e-6):

        # Data matrix
        self.X_ = X
        (nobs, nvars) = np.shape(X)
        self.n_obs_ = nobs
        self.nvars_ = nvars

        # Number of clusters
        self.k_ = k

        # PCA stopping input
        self.q_ = q

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
    def initialize_centroids(self, method='kmeans', Ci=None):

        '''This function will initialize the clusters centroids 
        using an avalibale method. Available methods are: uniform,
         k-means, random '''
        
        mlist = ["kmeans", "random", "uniform"]
        if method not in mlist:
            raise ValueError("Initialization method not recognized")
        
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
                self.C_[i,:] = np.mean(self.X_[self.labels==i,:], axis=0)

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

        return self
        
    def initialize_pca(self, n_start=2):

        '''This function will initialize the PCA objects in each cluster'''

        self.pca_ = []
        for i in range(self.k_):
            self.pca_.append(PCA(n_components=n_start))
            idps = np.where(self.labels==i)[0]
            self.pca_[i].fit(self.X_[idps,:])
        return self

    def fit(self, verbose=True):

        # Initialize global time
        st_global = datetime.now()

        # Initialize flag for convergence
        conv = False
        # Initialize iteration counter
        iter = 0

        while conv == False and iter < self.itmax_:

            # Monitor iteration time
            st_iter = datetime.now()

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
            if delta_centroids < self.ctol:
                conv = True
                print('Converged because of the centroids')

            # Check convergence on reconstruction error variance
            eps_rec_new = np.min(eps_matrix, axis=1)
            if iter == 0:
                self.eps_ = np.mean(eps_rec_new)
            else:
                delta_eps = (self.eps_ - eps_rec_new) / self.eps_
                if delta_eps < self.rtol_:
                    conv = True
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
            self.eps_    = eps_rec_new

            # Update iteration counter 
            iter += 1
            if iter == self.itmax_:
                print("Iterations reached maximum allowed number")

            # Time for iteration
            et_iter = datetime.now()
            dt_iter = st_iter - et_iter

            if verbose == True:
                print("Iteration: ", iter)
                print("Mean squared global reconstruction error = %E" % self.eps_ )
                print("Mean squared error in each cluster: ")
                print(rec_err_clust)
                print(f"Elapsed time for iteration: {dt_iter}")
        
        # Measure time at convergence
        et_global = datetime.now()
        dt_global = st_global - et_global
        print(f"Elapsed time for iteration: {dt_global} seconds")

        # Display information
        print("Obtained reconstruction error = %E", self.eps_)

        return labels_new, self










    

            
        







        

        




        

        

        

        



        



