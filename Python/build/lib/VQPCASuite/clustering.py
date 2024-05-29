import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, davies_bouldin_score, silhouette_score
import time
from datetime import datetime
import os
import h5py
from .preprocess import Scaler
from .metrics import ssim, psnr

class vqpca:

    def __init__(self, X, k=2, stopping_rule="variance", q=0.99, itmax=200, atol=1e-8, rtol=1e-8, ctol=1e-6):

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

    def fit(self, n_start=2, verbose=True, init_centroids='kmeans'):

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
        given, then the reconstructed data will be the original ones'''

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

    ### VQPCA evaluation ###
    def evaluate(self, score='ILPCA'):

        # ILPCA score evaluation
        if score == 'ILPCA':

            # Calculate squared reconstruction error in the clusters
            rec_err_clust = np.zeros(self.k_)
            for i in range(self.k_):
                U_scores = self.pca_[i].transform(self.X_[self.labels_==i])
                X_rec = self.pca_[i].inverse_transform(U_scores)
                rec_err_clust[i] = (np.mean(np.sum((self.X_[self.labels_==i] - X_rec)**2, axis=1)))

            # Initialize metric
            metric = 0.0
            for i in range(self.k_):
                # Reconstruction error in cluster i
                eps_i = rec_err_clust[i]
                # Initialize db
                db_iter = 0.0
                for j in range(self.k_):
                    # Reconstruction error in cluster j
                    eps_j = rec_err_clust[j]
                    # Merge cluster i and j
                    X_ij = np.vstack((self.X_[self.labels_==i,:], self.X_[self.labels_==j]))
                    # Perform PCA in the merged cluster
                    pca = PCA(n_components=self.q_)
                    pca.fit(X_ij)
                    # Reconstruct Xij
                    U_scores = pca.transform(X_ij)
                    X_rec    = pca.inverse_transform(U_scores)
                    # Reconstruction error of merged cluster
                    eps_ij = np.mean(np.sum((X_ij - X_rec)**2, axis=1))
                    # Get max between all the clusters pairs
                    db_iter = max(db_iter, (eps_i + eps_j)/eps_ij)

                metric += db_iter # Sum for the clusters
            
            # Average
            metric = metric / self.k_

        # If DB index was chosen
        elif score == "DB":
            metric = davies_bouldin_score(self.X_, self.labels_)

        elif score == "silhouette":
            metric = silhouette_score(self.X_, self.labels_)

        return metric
    

# ----------------- For data compression -------------------- #
import psutil
import logging

class compressor:

    # Default initializer
    def __init__(self, method='vqpca'):
        # Compression method
        self.method_ = method

    # Function to monitor resources
    def monitor_resources(self):
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        print(f"CPU usage: {cpu_usage}%")
        print(f"Memory usage: {memory_info.percent}%")

    # Function to compress the data
    def fit(self, X, q=0.99, k=5, scale=True, scaling='auto', verbose=False):
        # Monitor total execution time
        st = time.time()
        process = psutil.Process(os.getpid())
        memory_info_start = process.memory_info()
        rss_start = memory_info_start.rss / (1024 * 1024 * 1024)  # Convert to GB
        # Data pre-processing
        if scale:
            scaler = Scaler(method=scaling)
            Xscaled = scaler.fit_transform(X)
        else:
            Xscaled = X
        # Monitor resources
        if verbose:
            print("Pre-processing done")
            time.sleep(2)
            self.monitor_resources()
        # Initialize the compressor
        if self.method_ == 'vqpca':
            VQPCA = vqpca(Xscaled, k=k, stopping_rule="variance", q=q)
            # Fit VQPCA
            VQPCA.fit(verbose=False)
            # Monitor usage
            if verbose:
                print("Compression via VQPCA done")
                time.sleep(2)
                self.monitor_resources()
            # Get compressed data
            Xrec = VQPCA.reconstruct(Xscaled)
            if verbose:
                print("Reconstruction via VQPCA done")
                time.sleep(2)
                self.monitor_resources()
            # Update self
            self.Xrec_ = Xrec
            self.X_ = X
            # End time
            et = time.time()
            elapsed_time = et - st
            logging.info(f"Script finished in {elapsed_time} seconds")
            # End RAM
            memory_info_end = process.memory_info()
            rss_end = memory_info_end.rss / (1024 * 1024 * 1024)  # Convert to GB
            print(f"Memory usage: {rss_start:.2f} GB -> {rss_end:.2f} GB (RSS)")

        return Xrec
    
    def evaluate(self, custom_loss=False):

        '''This function will evaluate several parameters:
        the compression ratio, the loss in terms of MSE, NMSE,
        NRMSE, peak signal-to-noise-ratio, structural similarity index,
        for global and for each column of X'''

        # Global metrics
        g_mse = mean_squared_error(self.X_, self.Xrec_)
        # Structural Similarity Index (SSIM)
        g_ssim = ssim(self.X_, self.Xrec_)

        print("Global MSE = ", g_mse)
        print("Global SSIM = ", g_ssim)

        # Metrics for each feature
        l_mse = []
        l_ssim = []
        l_psnr = []
        for i in range(np.shape(self.X_)[1]):
            print(f" --- Evaluating feature {i} of {np.shape(self.X_)[1]} ---")
            l_mse.append(mean_squared_error(self.X_[:,i],self.Xrec_[:,i]))
            l_ssim.append(ssim(self.X_[:,i], self.Xrec_[:,i]))
            l_psnr.append(psnr(self.X_[:,i], self.Xrec_[:,i]))

        print("Evaluation completed for each feature")
        
        # Create a dictionary
        scores = {}
        scores['GlobalMSE'] = g_mse
        scores['GlobalSSIM'] = g_ssim
        scores["LocalMSE"] = l_mse
        scores["LocalSSIM"] = l_ssim
        scores["LocalPSNR"] = l_psnr

        return scores





    

            

        
        



    

        





        




                







            



    

    

        












    

            
        







        

        




        

        

        

        



        



