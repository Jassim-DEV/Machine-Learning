import numpy as np
import matplotlib.pyplot as pl

from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import *
from sklearn.cluster import AgglomerativeClustering

from pandas import *
from pandas.tools.plotting import scatter_matrix

#%matplotlib inline
#np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation


DataSet = read_csv('London_2014.csv', skipinitialspace=True)
#X = DataSet.iloc[:,[1,7]].values

f2 = DataSet['Mean TemperatureC'].values
f1 = DataSet['Mean Sea Level PressurehPa'].values
X = np.array(list(zip(f1, f2)))

#pl.title('Nuages de points')
#pl.xlabel('Mean Sea Level PressurehPa')
#pl.ylabel('Mean TemperatureC')
#pl.scatter(f1, f2, c='black', s=7)

#dimension des donnees
#print(DataSet.shape)
#statistiques descriptives
#print(DataSet.describe())
#graphique - croisement deux a deux des variables

#scatter_matrix(DataSet,figsize=(10,9))

## Dendograme  (figsize=(width, height))
#pl.figure
#pl.title("Dendograms") 
#dendrogram = dendrogram(linkage(X, method='ward'))
#  
#
### clusters 
#cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')  
#cluster.fit_predict(X)
#
##print(cluster.labels_)
#pl.figure()  
#pl.title('Clustering sur les conditions meteos')
#pl.xlabel('Mean Sea Level PressurehPa')
#pl.ylabel('Mean TemperatureC')
#pl.legend()
#pl.scatter(X[:,0],X[:,1], s = 15, c=cluster.labels_, cmap='rainbow')  
###

z = linkage(X, metric='cosine', method='complete') 
labels = fcluster(z, 0.1, criterion="distance")
pl.scatter(X[:, 0], X[:, 1], s= 15) 
pl.show() 