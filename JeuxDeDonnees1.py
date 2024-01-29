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

# data = arff.loadarff ( open ( "spherical_4_3.arff", 'r') )
# datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
# datanp = np.array(datanp)
# f0 = datanp[: , 0] # tous les elements de la premiere colonne
# f1 = datanp[: , 1] # tous les elements de la deuxieme colonne
# plt.scatter ( f0 , f1 , s = 8 )
# plt.title ( " Data spherical_4_3 " )
# plt.show ()

# data = arff.loadarff ( open ( "donut3.arff" , 'r') )
# datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
# datanp = np.array(datanp)
# f0 = datanp[: , 0] # tous les elements de la premiere colonne
# f1 = datanp[: , 1] # tous les elements de la deuxieme colonne
# plt.scatter ( f0 , f1 , s = 8 )
# plt.title ( " Data donut3 " )
# plt.show ()

# data = arff.loadarff ( open ( "smile1.arff" , 'r') )
# datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
# datanp = np.array(datanp)
# f0 = datanp[: , 0] # tous les elements de la premiere colonne
# f1 = datanp[: , 1] # tous les elements de la deuxieme colonne
# plt.scatter ( f0 , f1 , s = 8 )
# plt.title ( " Data smile1 " )
# plt.show ()

# data = arff.loadarff ( open ( "shapes.arff" , 'r') )
# datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
# datanp = np.array(datanp)
# f0 = datanp[: , 0] # tous les elements de la premiere colonne
# f1 = datanp[: , 1] # tous les elements de la deuxieme colonne
# plt.scatter ( f0 , f1 , s = 8 )
# plt.title ( " Data shapes " )
# plt.show ()

# data = arff.loadarff ( open ( "xclara.arff" , 'r') )
# datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
# datanp = np.array(datanp)
# f0 = datanp[: , 0] # tous les elements de la premiere colonne
# f1 = datanp[: , 1] # tous les elements de la deuxieme colonne
# plt.scatter ( f0 , f1 , s = 8 )
# plt.title ( " Data xclara " )
# plt.show ()

###############################################################################

##
#   Clustering k-Means
##


# # Select dataset 
# data = arff.loadarff ( open ( "donut3.arff" , 'r') )
# datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
# datanp = np.array(datanp)
# f0 = datanp[: , 0] # tous les elements de la premiere colonne
# f1 = datanp[: , 1] # tous les elements de la deuxieme colonne
# tps1 = time.time ()

# # Fixe un nombre de cluster
# k = 3
# model = cluster.KMeans( n_clusters =k , init='k-means++')
# model.fit( datanp )
# tps2 = time.time ()
# labels = model.labels_
# iteration = model.n_iter_

# #Plot labeled dataset
# plt.scatter( f0 , f1 , c = labels , s = 8 )
# plt.title ( " Donnees apres clustering Kmeans " )
# plt.show ()
# print ( " nb clusters = " ,k , " , nb iter = " , iteration  , " , runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )


# ##
# #   Métrique évaluation 
# ##

# silhouette_avg = silhouette_score(datanp, labels)
# print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
# davies = davies_bouldin_score(datanp, labels)
# print("For n_clusters =", k, "The average davies_score is :", davies)
# calinski = calinski_harabasz_score(datanp, labels)
# print("For n_clusters =", k, "The average calinski_score is :", calinski)
# print("\n")
# print()


###############################################################################

def Print_Dataset(dataset_name):
    data = arff.loadarff ( open ( dataset_name , 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
    datanp = np.array(datanp)
    f0 = datanp[: , 0] # tous les elements de la premiere colonne
    f1 = datanp[: , 1] # tous les elements de la deuxieme colonne
    plt.scatter ( f0 , f1 , s = 8 )
    plt.title (dataset_name)
    plt.show ()
    
def K_Means_Clustering(dataset_name,k):
    # Select dataset 
    data = arff.loadarff ( open ( dataset_name , 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
    datanp = np.array(datanp)
    f0 = datanp[: , 0] # tous les elements de la premiere colonne
    f1 = datanp[: , 1] # tous les elements de la deuxieme colonne
    tps1 = time.time ()
    # Fixe un nombre de cluster
    model = cluster.KMeans( n_clusters =k , init='k-means++')
    model.fit( datanp )
    tps2 = time.time ()
    labels = model.labels_
    iteration = model.n_iter_

    #Plot labeled dataset
    plt.scatter( f0 , f1 , c = labels , s = 8 )
    plt.title (f"{dataset_name} after clustering Kmeans, k = {k}" )
    plt.show ()
    print ( " nb clusters = " ,k , " , nb iter = " , iteration  , " , runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )


def K_Means_Clustering_silhouette(dataset_name,details_plot=False,k_max=10):

    # Select dataset 
    data = arff.loadarff ( open ( dataset_name , 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
    datanp = np.array(datanp)
    k_tab=[]
    score_tab=[]
    time_tab=[]
    best_silhouette_score=-1
    best_k=1
    
    for k in np.arange(2,k_max+1,1):
        tps1 = time.time ()
        model = cluster.KMeans( n_clusters =k , init='k-means++')
        model.fit( datanp )
        labels = model.labels_
        
        # plot cluster 
        if details_plot : K_Means_Clustering(dataset_name,k)

        # calcule silhouette metric
        silhouette_avg = silhouette_score(datanp, labels)
        tps2 = time.time ()
        k_tab.append(k)
        score_tab.append(silhouette_avg)
        time_tab.append(round (( tps2 - tps1 ) * 1000 , 2 ))
        print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg," , runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms ")
        if silhouette_avg > best_silhouette_score or k ==2 :
            best_silhouette_score = silhouette_avg
            best_k = k
        
    #Plot labeled dataset
    K_Means_Clustering(dataset_name,best_k)
    print ( " Best numnber of clustdaviesers = " ,best_k)
    plt.plot(k_tab,score_tab)
    plt.xlabel('k')
    plt.ylabel('silhouette score')
    plt.title("Variation of silhouette score when k increase ")
    plt.legend()
    plt.show()
    
    
def K_Means_Clustering_davies_bouldin(dataset_name,details_plot=False,k_max=10):

    # Select dataset 
    data = arff.loadarff ( open ( dataset_name , 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
    datanp = np.array(datanp)
    k_tab=[]
    score_tab=[]
    time_tab=[]
    best_davies_bouldin_score=1
    best_k=1
    
    for k in np.arange(2,k_max+1,1):
        tps1 = time.time ()
        model = cluster.KMeans( n_clusters =k , init='k-means++')
        model.fit( datanp )
        labels = model.labels_
        
        # plot cluster 
        if details_plot : K_Means_Clustering(dataset_name,k)

        # calcule davies metric
        davies = davies_bouldin_score(datanp, labels)
        tps2 = time.time ()
        k_tab.append(k)
        score_tab.append(davies)
        time_tab.append(round (( tps2 - tps1 ) * 1000 , 2 ))
        print("For n_clusters =", k, "The average davies_score is :", davies," , runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms ")
        if davies < best_davies_bouldin_score or k == 2:
            best_davies_bouldin_score = davies
            best_k = k
        
    #Plot labeled dataset
    K_Means_Clustering(dataset_name,best_k)
    print ( " Best numnber of clusters = " ,best_k)
    plt.plot(k_tab,score_tab)
    plt.xlabel('k')
    plt.ylabel('davies score')
    plt.title("Variation of davies score when k increase ")
    plt.legend()
    plt.show()

def K_Means_Clustering_calinski(dataset_name,details_plot=False,k_max=10):

    # Select dataset 
    data = arff.loadarff ( open ( dataset_name , 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
    datanp = np.array(datanp)
    k_tab=[]
    score_tab=[]
    time_tab=[]
    best_calinski_harabasz_score=-1
    best_k=1
    
    for k in np.arange(2,k_max+1,1):
        tps1 = time.time ()
        model = cluster.KMeans( n_clusters =k , init='k-means++')
        model.fit( datanp )
        labels = model.labels_
        
        # plot cluster 
        if details_plot : K_Means_Clustering(dataset_name,k)

        # calcule calinski metric
        calinski = calinski_harabasz_score(datanp, labels)
        tps2 = time.time ()
        k_tab.append(k)
        score_tab.append(calinski)
        time_tab.append(round (( tps2 - tps1 ) * 1000 , 2 ))
        print("For n_clusters =", k, "The average calinski_score is :", calinski," , runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms ")
        if calinski > best_calinski_harabasz_score or k ==2 :
            best_calinski_harabasz_score = calinski
            best_k = k
        
    #Plot labeled dataset
    K_Means_Clustering(dataset_name,best_k)
    print ( " Best numnber of clustdaviesers = " ,best_k)
    plt.plot(k_tab,score_tab)
    plt.xlabel('k')
    plt.ylabel('calinski score')
    plt.title("Variation of calinski score when k increase ")
    plt.legend()
    plt.show()

################################################################################################


# Print_Dataset("smile1.arff")
# K_Means_Clustering("shapes.arff",4)
# K_Means_Clustering("donut3.arff",3)
# K_Means_Clustering("smile1.arff",4)
# K_Means_Clustering("spherical_4_3.arff",4)


def metric_analyse(name,details_plot=False):
    K_Means_Clustering_silhouette(name,details_plot)
    K_Means_Clustering_davies_bouldin(name,details_plot)
    K_Means_Clustering_calinski(name,details_plot)

# Working well with Kmeans
# metric_analyse("xclara.arff",False)
# metric_analyse("spherical_4_3.arff",False)

# # Not working well with Kmeans
# metric_analyse("smile1.arff",False)
# metric_analyse("donut3.arff",False)
# metric_analyse("3-spiral.arff",False)

# Working well for the only two first metri. Why ????
#metric_analyse("shapes.arff",False)
























