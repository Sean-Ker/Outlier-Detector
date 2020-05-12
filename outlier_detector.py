import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

'''
A function that detects outliers, where k is a tandard deviation threshold hyperparameter preferablly (2, 2.5, 3).
The algo could handle multivariable data frames with any number of features d.
For that manner, it first reduces the dimensionality to 2 using PCA, makes sure that the matrix is positive definite and calculates the Mahalanobis Distance with a threshold value.
Returns a series of n rows back.
'''
def outlier_detector(data, k=2.5):
    # Calculate Principal Component Analysis
    pca = PCA(n_components=data.shape[1], svd_solver='full')
    df = pd.DataFrame(pca.fit_transform(
        data), index=data.index, columns=data.columns)

    # Calculate covariance and its inverse matrices
    cov_matrix = np.cov(df.values, rowvar=False)
    inv_cov = np.linalg.inv(cov_matrix)
    mean = df.values.mean(axis=0)

    # Check matrices are positive definite: https://en.wikipedia.org/wiki/Definiteness_of_a_matrix
    assert is_pos_def(cov_matrix) and is_pos_def(inv_cov)

    # Calculate Mahalanobis Distance https://en.wikipedia.org/wiki/Mahalanobis_distance
    md = mahalanobis_dist(inv_cov, mean, df.values, verbose=False)
    threshold = np.mean(md) * k

    # res = pd.DataFrame(index=data.index,columns=data.columns)

    return data[md > threshold]


# https://www.youtube.com/watch?v=spNpfmWZBmg&t=0s
def mahalanobis_dist(inv_cov_matrix, mean_distr, data, verbose=False):
    diff = data - mean_distr
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_cov_matrix).dot(diff[i])))
    return np.array(md)

# Check that matrix is positive definite
def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

