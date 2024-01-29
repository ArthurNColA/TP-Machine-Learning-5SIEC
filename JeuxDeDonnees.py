#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:59:41 2024

@author: nicola
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff

import time
from sklearn import cluster

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score


path = './clustering-benchmark/src/main/resources/datasets/artificial/'

#path = '/Users/arthurnicola/Desktop/TP_clustering/clustering-benchmark/src/main/resources/datasets/artificial/'

# Parser un fichier de donnees au format arff
# data est un tableau d ’ exemples avec pour chacun
# la liste des valeurs des features
#
# Dans les jeux de donnees consideres :
# il y a 2 features ( dimension 2 )
# Ex : [[ - 0 . 499261 , -0 . 0612356 ] ,
# [ - 1 . 51369 , 0 . 265446 ] ,
# [ - 1 . 60321 , 0 . 362039 ] , .....
# ]
#
# Note : chaque exemple du jeu de donnees contient aussi un
# numero de cluster . On retire cette information



##
#   Visualisation des jeux de données 
##


# databrut = arff.loadarff ( open ( path + "donut3.arff", 'r') )
# datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
# f0=[x[0] for x in datanp]
# f1=[x[1] for x in datanp]
# plt.scatter ( f0 , f1 , s = 8 )
# plt.title ( " Data donut3 " )
# plt.show ()

# databrut = arff.loadarff ( open ( path + "smile1.arff", 'r') )
# datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
# f0=[x[0] for x in datanp]
# f1=[x[1] for x in datanp]
# plt.scatter ( f0 , f1 , s = 8 )
# plt.title ( " Data smile1 " )
# plt.show ()

# databrut = arff.loadarff ( open ( path + "shapes.arff", 'r') )
# datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
# f0=[x[0] for x in datanp]
# f1=[x[1] for x in datanp]
# plt.scatter ( f0 , f1 , s = 8 )
# plt.title ( " Data shapes " )
# plt.show ()

# databrut = arff.loadarff ( open ( path + "xclara.arff", 'r') )
# datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
# f0=[x[0] for x in datanp]
# f1=[x[1] for x in datanp]
# plt.scatter ( f0 , f1 , s = 8 )
# plt.title ( " Data xclara " )
# plt.show ()

###############################################################################

##
#   Clustering k-Means
##

print(" Appel KMeans pour une valeur fixee de k ")

# Select dataset 

data = arff.loadarff ( open ( "xclara.arff" , 'r') )
datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
datanp = np.array(datanp)
f0 = datanp[: , 0] # tous les elements de la premiere colonne
f1 = datanp[: , 1] # tous les elements de la deuxieme colonne
plt.scatter( f0, f1 , s = 8 )
plt.title( " Donnees xlara " )
plt.show()

# Fixe un nombre de cluster
k = 4
tps1 = time . time ()
model = cluster.KMeans( n_clusters =k , init='k-means++')
model.fit( datanp )
tps2 = time.time ()
labels = model.labels_
iteration = model.n_iter_

#Plot labeled dataset
plt.scatter( f0 , f1 , c = labels , s = 8 )
plt.title ( " Donnees apres clustering Kmeans " )
plt.show ()
print ( " nb clusters = " ,k , " , nb iter = " , iteration  , " , runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

##
#   Métrique évaluation 
##

silhouette_avg = silhouette_score(datanp, labels)
print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
davies = davies_bouldin_score(datanp, labels)
print("For n_clusters =", k, "The average davies_score is :", davies)
calinski = calinski_harabasz_score(datanp, labels)
print("For n_clusters =", k, "The average calinski_score is :", calinski)
print("\n")
print()


###############################################################################

def Print_Dataset(dataset_name):
    databrut = arff.loadarff ( open ( path + dataset_name, 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
    f0=[x[0] for x in datanp]
    f1=[x[1] for x in datanp]
    plt.scatter ( f0 , f1 , s = 8 )
    plt.title (dataset_name)
    plt.show ()
    
def K_Means_Clustering(dataset_name,k):
    # Select dataset 
    databrut = arff.loadarff ( open ( path + dataset_name, 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
    f0=[x[0] for x in datanp]
    f1=[x[1] for x in datanp]
    tps1 = time.time ()
    # Fixe un nombre de cluster
    model = cluster.KMeans( n_clusters =k , init='k-means++')
    model.fit( datanp )
    tps2 = time.time ()
    labels = model.labels_
    iteration = model.n_iter_

    #Plot labeled dataset
    plt.scatter( f0 , f1 , c = labels , s = 8 )
    plt.title (f"Data after clustering Kmeans, k = {k}" )
    plt.show ()
    print ( " nb clusters = " ,k , " , nb iter = " , iteration  , " , runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )


def K_Means_Clustering_silhouette(dataset_name,details_plot=False):

    # Select dataset 
    databrut = arff.loadarff ( open ( path + dataset_name, 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
    best_silhouette_score=-1
    best_k=1
    
    for k in np.arange(2,10,1):
        model = cluster.KMeans( n_clusters =k , init='k-means++')
        model.fit( datanp )
        labels = model.labels_
        
        # plot cluster 
        if details_plot : K_Means_Clustering(dataset_name,k)

        # calcule silhouette metric
        silhouette_avg = silhouette_score(datanp, labels)
        print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
        if silhouette_avg > best_silhouette_score or k ==2 :
            best_silhouette_score = silhouette_avg
            best_k = k
        
    #Plot labeled dataset
    K_Means_Clustering(dataset_name,best_k)
    print ( " Best numnber of clustdaviesers = " ,best_k)
    
def K_Means_Clustering_davies_bouldin(dataset_name,details_plot=False):

    # Select dataset 
    databrut = arff.loadarff ( open ( path + dataset_name, 'kr') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
    best_davies_bouldin_score=1
    best_k=1
    
    for k in np.arange(2,10,1):
        model = cluster.KMeans( n_clusters =k , init='k-means++')
        model.fit( datanp )
        labels = model.labels_
        
        # plot cluster 
        if details_plot : K_Means_Clustering(dataset_name,k)

        # calcule davies metric
        davies = davies_bouldin_score(datanp, labels)
        print("For n_clusters =", k, "The average davies_score is :", davies)
        if davies < best_davies_bouldin_score or k == 2:
            best_davies_bouldin_score = davies
            best_k = k
        
    #Plot labeled dataset
    K_Means_Clustering(dataset_name,best_k)
    print ( " Best numnber of clusters = " ,best_k)

def K_Means_Clustering_calinski(dataset_name,details_plot=False):

    # Select dataset 
    databrut = arff.loadarff ( open ( path + dataset_name, 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
    best_calinski_harabasz_score=-1
    best_k=1
    
    for k in np.arange(2,10,1):
        model = cluster.KMeans( n_clusters =k , init='k-means++')
        model.fit( datanp )
        labels = model.labels_
        
        # plot cluster 
        if details_plot : K_Means_Clustering(dataset_name,k)

        # calcule calinski metric
        calinski = calinski_harabasz_score(datanp, labels)
        print("For n_clusters =", k, "The average calinski_score is :", calinski)
        if calinski > best_calinski_harabasz_score or k ==2 :
            best_calinski_harabasz_score = calinski
            best_k = k
        
    #Plot labeled dataset
    K_Means_Clustering(dataset_name,best_k)
    print ( " Best numnber of clustdaviesers = " ,best_k)

################################################################################################

## f1
# Print_Dataset("smile1.arff")

## f2
# K_Means_Clustering("smile1.arff",4)

##f3
# K_Means_Clustering_silhouette("xclara.arff")
# K_Means_Clustering_silhouette("smile1.arff",True)
#f4
#K_Means_Clustering_davies_bouldin("xclara.arff")
K_Means_Clustering_calinski("xclara.arff",True)



# databrut = arff.loadarff ( open ( path + "spherical_4_3.arff", 'r') )
# datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
# f0=[x[0] for x in datanp]
# f1=[x[1] for x in datanp]
# plt.scatter ( f0 , f1 , s = 8 )
# plt.title ( " Donnees spherical_4_3 " )
# plt.show ()

# ilhouette_avg = silhouette_score(datanp, labels)
# print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)

# davies = davies_bouldin_score(datanp, labels)
# print("For n_clusters =", k, "The average davies_score is :", davies)

# calinski = calinski_harabasz_score(datanp, labels)
# print("For n_clusters =", k, "The average calinski_score is :", calinski)
# print("\n")
# print()


# print(" Appel KMeans pour une valeur fixee de k ")
# tps1 = time.time ()
# k = 4
# model = cluster.KMeans( n_clusters =k , init='k-means++')
# model.fit( datanp )
# tps2 = time.time ()
# labels = model.labels_
# iteration = model.n_iter_
# plt.scatter( f0 , f1 , c = labels , s = 8 )
# plt.title ( " Donnees apres clustering Kmeans " )
# plt.show ()
# print ( " nb clusters = " ,k , " , nb iter = " , iteration  , " , runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

# # The silhouette_score gives the average value for all the samples.
# # This gives a perspective into the density and separation of the formed
# # clusters
# silhouette_avg = silhouette_score(datanp, labels)
# print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)

# davies = davies_bouldin_score(datanp, labels)
# print("For n_clusters =", k, "The average davies_score is :", davies)

# calinski = calinski_harabasz_score(datanp, labels)
# print("For n_clusters =", k, "The average calinski_score is :", calinski)
# print("\n")





# databrut = arff.loadarff ( open ( path + "3-spiral.arff", 'r') )
# datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in databrut [ 0 ] ]
# # Affichage en 2D
# # Extraire chaque valeur de features pour en faire une liste
# # Ex pour f0 = [ - 0 . 499261 , -1 . 51369 , -1 . 60321 , ...]
# # Ex pour f1 = [ - 0 . 0612356 , 0 . 265446 , 0 . 362039 , ...]
# #f0 = datanp [ : ,0 ] # tous les elements de la premiere colonne
# #f1 = datanp [ : ,1 ] # tous les elements de la deuxieme colonne
# f0=[x[0] for x in datanp]
# f1=[x[1] for x in datanp]
# plt.scatter ( f0 , f1 , s = 8 )
# plt.title ( " Donnees initiales " )
# plt.show ()

# print(" Appel KMeans pour une valeur fixee de k ")
# tps1 = time.time ()
# k = 3
# model = cluster.KMeans( n_clusters =k , init='k-meansk++')
# model.fit( datanp )
# tps2 = time.time ()
# labels = model.labels_
# iteration = model.n_iter_
# plt.scatter( f0 , f1 , c = labels , s = 8 )
# plt.title ( " Donnees apres clustering Kmeans " )
# plt.show ()
# print ( " nb clusters = " ,k , " , nb iter = " , iteration  , " , runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

# # The silhouette_score gives the average value for all the samples.
# # This gives a perspective into the density and separation of the formed
# # clusters
# silhouette_avg = silhouette_score(datanp, labels)
# print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)

# davies = davies_bouldin_score(datanp, labels)
# print("For n_clusters =", k, "The average davies_score is :", davies)

# calinski = calinski_harabasz_score(datanp, labels)
# print("For n_clusters =", k, "The average calinski_score is :", calinski)
# print("\n")
















