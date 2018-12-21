import numpy as np
import matplotlib.pyplot as pl

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pandas import *



london = read_csv('London_2014.csv')
london = read_csv('London_2014.csv', skipinitialspace=True)
london = london.rename(columns={'WindDirDegrees<br />' : 'WindDirDegrees'})
london['WindDirDegrees'] = london['WindDirDegrees'].str.rstrip('<br />')
london['Events'] = london['Events'].fillna('')
london[london['Events'].isnull()]
london.dropna()
london['WindDirDegrees'] = london['WindDirDegrees'].astype('int64') 
london['GMT'] = to_datetime(london['GMT'])

 
f1 = london['Mean Sea Level PressurehPa'].values
f2 = london['Mean TemperatureC'].values

X = np.array(list(zip(f1, f2)))
#X = np.array(list(zip(f1, f2)))
#X = london.iloc[:,[8,14]].values
#pl.scatter(f1, f2, c='black', s=7)

#pl.figure()
#X = london.iloc[:,[1,7]].values
#pl.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow') 


# Elbow Method

I = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    I.append(kmeans.inertia_)
    
pl.figure()
pl.plot(range(1, 11), I)
pl.title('The Elbow Method')
pl.xlabel('Number of Clusters')
pl.ylabel('WCSS')
pl.show()


# Algorithme du Kmeans

kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)

#visualisation des clusters

pl.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 25, c = 'red',     label = 'Cluster 1')
pl.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 25, c = 'blue',    label = 'Cluster 2')
pl.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 25, c = 'green',   label = 'Cluster 3')
#pl.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 25, c = 'cyan',    label = 'Cluster 4')
#pl.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 25, c = 'magenta', label = 'Cluster 5')
pl.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 150, label = 'Centroids', marker = 'x')
pl.title('K-Means Clustering sur les conditions meteos')
pl.xlabel('Mean Sea Level PressurehPa')
pl.ylabel('Mean TemperatureC')
pl.legend()
pl.show()

