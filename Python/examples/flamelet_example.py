#%%
'''In this example, we perform a VQPCA clustering evaluation with several 
number of clusters and variance cutoffs. The clustering solutions
will be evaluated using the Ilpca index availble in the literature'''

import numpy as np
import pandas as pd
from VQPCASuite.clustering import vqpca
from VQPCASuite.preprocess import Scaler
import sys

# ---------- 0) Data import section ---------- #

# Load the hydrogen flamelet dataset
# Import the flamelet dataset
path_to_files = '/Users/matteosavarese/Desktop/Dottorato/Datasets/hydrogen-air-flamelet/'
filename = 'STEADY-clustered-flamelet-H2-'
# Import state space
state_space = pd.read_csv(path_to_files + filename + 'state-space.csv', header=None)
# Import state space sources
state_space_sources = pd.read_csv(path_to_files + filename + 'state-space-sources.csv', header=None)
# Import mixture fraction
mixture_fraction = pd.read_csv(path_to_files + filename + 'mixture-fraction.csv', header=None)
# Import heat release rate
heat_release_rate = pd.read_csv(path_to_files + filename + 'heat-release-rate.csv', header=None)
# Import dissipation rate 
dissipation_rates = pd.read_csv(path_to_files + filename + 'dissipation-rates.csv', header=None)
# Import state space names as a list of strings
state_space_names = pd.read_csv(path_to_files + filename + 'state-space-names.csv', header=None)
# Assign column names to the dataframes
state_space.columns = state_space_names.iloc[:,0]
state_space.head()
# Assign column names to the sources dataframe
state_space_sources.columns = state_space_names.iloc[:,0]
# Print column names
print(state_space.columns)
# %%

# ---------- 1) Data pre processing ---------- #
scaling = 'auto'                    # pareto, vast, minmax, level, auto, no
scaler = Scaler(method=scaling)     # initalize scaler
# Get scaled data
X = state_space.values
X_scaled = scaler.fit_transform(X)

# %%

# ---------- 2) Perform VQPCA ---------- #
qspan = np.array([0.90, 0.95, 0.99])
kspan = np.array([2, 3, 4, 5, 6, 7])
ids = np.zeros((len(qspan), len(kspan)))
for i in range(len(qspan)):
    for j in range(len(kspan)):

        # Initialize vqpca object
        model = vqpca(X_scaled, k=kspan[j], q=qspan[i])
        # Fit object
        model.fit()
        # Evaluate solution
        value = model.evaluate()
        # Update array of solutions
        ids[i,j] = value

# %% 

# ---------- 3) Post processing and plots ---------- #
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4,3))
im = ax.imshow(ids, cmap="RdBu_r")
# Add colorbar
cb = fig.colorbar(im, ax=ax)
cb.set_label('$I_{LPCA}$')
# Modify axis labels
ax.set_xticks(np.arange(0, len(kspan)))
ax.set_yticks(np.arange(0, len(qspan)))
ax.set_xticklabels(kspan)
ax.set_yticklabels(qspan)
# Add labels
ax.set_xlabel('number of cluster')
ax.set_ylabel('retained variance')
fig.tight_layout()


# %%

# Data compression example
from VQPCASuite.clustering import compressor

# Compress data
comp = compressor()
comp.fit(X_scaled, q=0.9, verbose=True, scale=False)

# Get scores
U = comp.get_compressed_data()

# Get total size of compressed data
s = 0
for i in range(comp.k_):
    s += sys.getsizeof(U[i])

s0 = sys.getsizeof(X_scaled)

print("Compression ratio = ", s0/s)
# %%
