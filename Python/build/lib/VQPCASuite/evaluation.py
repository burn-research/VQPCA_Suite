import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score


### VQPCA evaluation ###
def evaluate(X, labels, q=0.99, score='ILPCA'):

    '''This function is used to perform clustering evaluation using a selected method.
    
    Inputs:
    X (2D NumPy array): data matrix
    labels (1D NumPy array): array of cluster labels
    q (float): variance threshold or number of components (default=0.99)
    score (str): method for evaluation, available are 'ILPCA', 'DB', 'silhouette' (default ILPCA)
    
    Output:
    metric (float): value of the final averaged index'''

    # Check number of observations is coherent
    if np.shape(X)[0] != len(labels):
        raise ValueError("Number of observations of X and labels must be the same")

    # Number of clusters
    k = np.max(labels)+1

    # ILPCA score evaluation
    if score == 'ILPCA':

        # Perform PCA in each cluster
        pca_ = []
        for i in range(k):
            # Initialize PCA
            pca = PCA(n_components=q)
            pca.fit(X[labels==i,:])
            pca_.append(pca)

        # Calculate squared reconstruction error in the clusters
        rec_err_clust = np.zeros(k)
        for i in range(k):
            U_scores = pca_[i].transform(X[labels==i])
            X_rec = pca_[i].inverse_transform(U_scores)
            rec_err_clust[i] = (np.mean(np.sum((X[labels==i] - X_rec)**2, axis=1)))

        # Initialize metric
        metric = 0.0
        for i in range(k):
            # Reconstruction error in cluster i
            eps_i = rec_err_clust[i]
            # Initialize db
            db_iter = 0.0
            for j in range(k):
                if j != i:
                    # Reconstruction error in cluster j
                    eps_j = rec_err_clust[j]
                    # Merge cluster i and j
                    X_ij = np.vstack((X[labels==i,:], X[labels==j,:]))
                    # Perform PCA in the merged cluster
                    pca = PCA(n_components=q)
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
        metric = metric / k

    # If DB index was chosen
    elif score == "DB":
        metric = davies_bouldin_score(X, labels)

    elif score == "silhouette":
        metric = silhouette_score(X, labels)

    return metric