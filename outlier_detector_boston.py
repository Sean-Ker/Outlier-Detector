# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#Import the libraries
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import load_boston,load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

#Load the data
boston = load_boston()

#Find features and target
x = boston.data
y = boston.target

#Find the dic keys
print(boston.keys())


# %%
#find features name
columns = boston.feature_names
columns


# %%
#Create dataframe
boston_data = pd.DataFrame(boston.data)
boston_data.columns = columns
print(boston_data.shape)
boston_data['target'] = boston.target
boston_data


# %%
plt.figure(figsize= (4,4), dpi=100)
sns.heatmap(boston_data.corr(),cmap='hot')


# %%
def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)

g = sns.PairGrid(boston_data.iloc[:,5:8], aspect=1.4, diag_sharey=False)
g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
g.map_diag(sns.distplot, kde_kws={'color': 'black'})
g.map_upper(corrdot)


# %%
boston_data


# %%
x='INDUS'
y='TAX'
boston_df = boston_data[[x,y]]
boston_df


# %%
boston_df.corr()


# %%
#Multivariate outlier analysis
fig, ax = plt.subplots(figsize=(11,8.5))
ax.scatter(boston_df[x], boston_df[y])
ax.set_xlabel(x)
ax.set_ylabel(y)


# %%
pca = PCA(n_components=boston_df.shape[1], svd_solver= 'full')
df = pd.DataFrame(pca.fit_transform(boston_df),index=boston_df.index,columns=boston_df.columns) #[i.lower()+'_pca' for i in boston_df.columns]
plt.scatter(x=df[x],y=df[y])
plt.title('PCA Graph')


# %%
def cov_matrix(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            return covariance_matrix, inv_covariance_matrix
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")

def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    return np.array(md)

def MD_detectOutliers(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    outliers = []
    for i in range(len(dist)):
        if dist[i] >= threshold:
            outliers.append(i)  # index of the outlier
    return np.array(outliers)

def MD_threshold(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    return threshold

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


# %%
cov_matrix = np.cov(df.values,rowvar=False)
inv_cov = np.linalg.inv(cov_matrix)
mean = df.values.mean(axis=0)

# Check matrices are positive definite:https://en.wikipedia.org/wiki/Definiteness_of_a_matrix 
assert is_pos_def(cov_matrix) and is_pos_def(inv_cov)
# Check matrices are invereses
np.matmul(cov_matrix,inv_cov).astype(np.float16)


# %%
md = MahalanobisDist(inv_cov, mean, df.values, verbose=False)
threshold = 2#MD_threshold(md, extreme = False)
print("Threshold: "+ str(threshold))
plt.hist(list(md),bins=40)
plt.show()


# %%
plt.figure()
sns.distplot(np.square(md), bins = 40, kde= False)
plt.xlim([0.0,15])

plt.figure()
sns.distplot(md,
             bins = 40, 
             kde= True, 
            color = 'green')
# plt.xlim([0.0,5])
plt.xlabel('Mahalanobis dist')


# %%
# classify what data is an outlier  
# boston_df['anomaly'] = df['anomaly'] = md>threshold
# boston_df[boston_df.anomaly]
len(boston_df[md>threshold])


# %%
plt.figure(figsize=(10,8))
plt.scatter(boston_df[x][md<=threshold], boston_df[y][md<=threshold])
plt.scatter(boston_df[x][md>threshold], boston_df[y][md>threshold])
plt.show()

plt.figure(figsize=(10,8))
plt.scatter(df[x][md<=threshold], df[y][md<=threshold])
plt.scatter(df[x][md>threshold], df[y][md>threshold])
# plt.plot([threshold]*len(df[x]_pca),df.indus_pca,colour)

