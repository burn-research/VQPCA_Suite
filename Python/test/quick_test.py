import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from VQPCASuite.preprocess import *
from VQPCASuite.clustering import *

# Import test data
datafolder = "../../TestData/hydrogen-air-flamelet/"
filename = datafolder+"STEADY-clustered-flamelet-H2-state-space.csv"
sourcename = datafolder+"STEADY-clustered-flamelet-H2-state-space-sources.csv"

# Read data
dt = pd.read_csv(filename, header=None)
X  = dt.values
dp = pd.read_csv(sourcename, header=None)
Y = dp.values

# Preprocess
scaler = Scaler(method="auto")
Xs     = scaler.fit_transform(X)
Ys     = scaler.transform(Y, center=False)
scaler_source = Scaler(method="range")
Yss = scaler_source.fit_transform(Ys)

# Fit VQPCA
print("========= Checking VQPCA =========")
model = vqpca(Xs)
model.fit(k=2, q=0.99)

# Fit vqpls
print("========= Checking VQPLS =========")
model = vqpls()
model.fit(Xs, Yss, init='random', n_clusters=7, n_components=5)

print("Test done.")