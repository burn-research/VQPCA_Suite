# ----------------- For data compression -------------------- #
import psutil
import logging
from .compression import vqpca
from .preprocess import *
import time
import os
from sklearn.metrics import mean_squared_error, davies_bouldin_score, silhouette_score
from .metrics import ssim, psnr


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
            self.scaler_ = scaler
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

            # Reconstruct data
            Xrec = VQPCA.reconstruct(Xscaled)
            if verbose:
                print("Reconstruction via VQPCA done")
                time.sleep(2)
                self.monitor_resources()

            # End time
            et = time.time()
            elapsed_time = et - st
            logging.info(f"Script finished in {elapsed_time} seconds")

            # End RAM
            memory_info_end = process.memory_info()
            rss_end = memory_info_end.rss / (1024 * 1024 * 1024)  # Convert to GB
            print(f"Memory usage: {rss_start:.2f} GB -> {rss_end:.2f} GB (RSS)")

            # Update attributes
            self.k_ = k
            self.q_ = q

            # Get components
            components = VQPCA.get_components()
            self.A_ = components

            # Update self
            self.Xrec_ = Xrec
            self.X_ = X
            self.labels_ = VQPCA.labels_

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
    
    def get_compressed_data(self):

        # Initialize list of scores
        Uscores = []

        for i in range(self.k_):
            
            # Get local PCA
            A = self.A_[i]

            # Get scores
            U = self.X_[self.labels_==i] @ A
            Uscores.append(U)

        return Uscores