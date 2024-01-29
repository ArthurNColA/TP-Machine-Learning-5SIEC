#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:16:54 2024

@author: nicola, beneito
"""
import numpy as np

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from scipy.io import arff

import time
from sklearn import cluster

from sklearn.metrics import silhouette_score

def Print_Dataset(dataset_name):
    data = arff.loadarff ( open ( dataset_name , 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
    datanp = np.array(datanp)
    f0 = datanp[: , 0] # tous les elements de la premiere colonne
    f1 = datanp[: , 1] # tous les elements de la deuxieme colonne
    plt.scatter ( f0 , f1 , s = 8 )
    plt.title (dataset_name)
    plt.show ()
    
def Print_Dendogramme(dataset_name):
    # Select dataset 
    data = arff.loadarff ( open ( dataset_name, 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
    datanp = np.array(datanp)
    #f0 = datanp[: , 0] # tous les elements de la premiere colonne
    #f1 = datanp[: , 1] # tous les elements de la deuxieme colonne

    # Donnees dans datanp
    linked_mat = shc.linkage ( datanp , 'single')
    plt.figure ( figsize = ( 12 , 12 ) )
    shc.dendrogram ( linked_mat,orientation = 'top' , distance_sort = 'descending' , show_leaf_counts = False )
    plt.title ( f" Dendogramme : {dataset_name}" )
    plt.show ()
    
# Print_Dendogramme("smile1.arff")
# Print_Dendogramme("donut3.arff")
# Print_Dendogramme("shapes.arff")
# Print_Dendogramme("3-spiral.arff")


##########################################

##   
#   Select dataset 
##
data = arff.loadarff ( open ( "donut3.arff", 'r') )
datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
datanp = np.array(datanp)
f0 = datanp[: , 0] # tous les elements de la premiere colonne
f1 = datanp[: , 1] # tous les elements de la deuxieme colonne

##
#   Dendogramme
##
# Donnees dans datanp
# print ( " Dendrogramme ’ single ’ donnees initiales " )
# linked_mat = shc.linkage ( datanp , 'single')
# plt.figure ( figsize = ( 12 , 12 ) )
# shc.dendrogram ( linked_mat,orientation = 'top' , distance_sort = 'descending' , show_leaf_counts = False )
# plt.title ( " Dendogramme " )
# plt.show ()

##
#  Clustering when we set distance_threshold ( 0 ensures we compute the full tree )
##
# tps1 = time.time ()
# model = cluster.AgglomerativeClustering ( distance_threshold = 0.02 , linkage = 'single' , n_clusters = None )
# model = model.fit ( datanp )
# tps2 = time.time ()
# labels = model.labels_
# k = model.n_clusters_
# leaves = model.n_leaves_


# # Affichage clustering
# plt.scatter ( f0 , f1 , c = labels , s = 8 )
# plt.title ( f" Resultat du clustering en choisissant la distance : nb clusters = {k}" )
# plt.show ()
# print ( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

# ###
# #   Métrique s'évaluation
# ###
# silhouette_avg = silhouette_score(datanp, labels)
# print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)


##
# Clustering when we set the number of clusters
##
# k = 3
# tps1 = time.time ()
# model = cluster.AgglomerativeClustering ( linkage = 'single' , n_clusters = k )
# model = model.fit ( datanp )
# tps2 = time.time ()
# labels = model.labels_
# kres = model.n_clusters_
# leaves = model.n_leaves_

# # Affichage clustering
# plt.scatter ( f0 , f1 , c = labels , s = 8 )
# plt.title ( f"Resultat du clustering en choisissant le nombre de clusters : nb clusters = {k}" )
# plt.show ()
# print ( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
# print (" kres = " , kres)

####################################################################################

#  Clustering hiérarchique en utilisant une limite sur le seuil de distance        #
    
####################################################################################



def aggloomeratif_cluster_distance(name,d_max,d_step):
    
    data = arff.loadarff ( open ( name, 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
    datanp = np.array(datanp)
    f0 = datanp[: , 0] # tous les elements de la premiere colonne
    f1 = datanp[: , 1] # tous les elements de la deuxieme colonne
    link=['single','ward','complete','average']
    # Donnees dans datanp
    linked_mat = shc.linkage ( datanp , 'single')
    plt.figure ( figsize = ( 12 , 12 ) )
    shc.dendrogram ( linked_mat,orientation = 'top' , distance_sort = 'descending' , show_leaf_counts = False )
    plt.title ( " Dendogramme " )
    plt.show ()
    
    for l in link :
        cluster_f = True
        print()
        for d in np.arange(0,d_max,d_step):
            if cluster_f :
                # set distance_threshold ( 0 ensures we compute the full tree )
                tps1 = time.time ()
                model = cluster.AgglomerativeClustering ( distance_threshold = d , linkage = l , n_clusters = None )
                model = model.fit ( datanp )
                tps2 = time.time ()
                labels = model.labels_
                k = model.n_clusters_
                leaves = model.n_leaves_
                    
                # Affichage clustering
                plt.scatter ( f0 , f1 , c = labels , s = 8 )
                plt.title ( f" nb clusters = {k}, linkage = {l}, distance_threshold = {d}" )
                plt.show ()
                print ( " linkage = " ,l , " nb clusters = " ,k , "distance threshold = ",d , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
                if k == 1 : 
                    cluster_f = False

    
   
                
### Test : aggloomeratif_cluster()

##
#   smile1
##
# linkage =  single  nb clusters =  4 distance threshold =  0.03  , nb feuilles =  1000  runtime =  5.56  ms 
# linkage =  single  nb clusters =  4 distance threshold =  0.04  , nb feuilles =  1000  runtime =  5.14  ms 
# linkage =  single  nb clusters =  4 distance threshold =  0.05  , nb feuilles =  1000  runtime =  5.54  ms
# linkage =  single  nb clusters =  4 distance threshold =  0.06  , nb feuilles =  1000  runtime =  6.01  ms 
# linkage =  single  nb clusters =  4 distance threshold =  0.07  , nb feuilles =  1000  runtime =  5.56  ms 
# linkage =  single  nb clusters =  4 distance threshold =  0.08  , nb feuilles =  1000  runtime =  5.58  ms 
# linkage =  single  nb clusters =  4 distance threshold =  0.09  , nb feuilles =  1000  runtime =  5.56  ms 
# linkage =  single  nb clusters =  4 distance threshold =  0.1  , nb feuilles =  1000  runtime =  5.41  ms 
# linkage =  single  nb clusters =  4 distance threshold =  0.11  , nb feuilles =  1000  runtime =  5.18  ms 
# linkage =  single  nb clusters =  4 distance threshold =  0.12  , nb feuilles =  1000  runtime =  5.41  ms 
# linkage =  single  nb clusters =  4 distance threshold =  0.13  , nb feuilles =  1000  runtime =  5.31  ms 
#aggloomeratif_cluster_distance("smile1.arff",0.1,0.01) 

##
#   donut3
##
# linkage =  single  nb clusters =  3 distance threshold =  0.02  , nb feuilles =  999  runtime =  7.3  ms 
#aggloomeratif_cluster_distance("donut3.arff",0.5,0.01) 

##
#   3-spiral
##
# linkage =  single  nb clusters =  3 distance threshold =  1.5  , nb feuilles =  312  runtime =  1.71  ms 
# linkage =  single  nb clusters =  3 distance threshold =  2.0  , nb feuilles =  312  runtime =  1.65  ms 
# linkage =  single  nb clusters =  3 distance threshold =  2.5  , nb feuilles =  312  runtime =  2.97  ms 
# linkage =  single  nb clusters =  3 distance threshold =  3.0  , nb feuilles =  312  runtime =  1.77  ms 
# linkage =  single  nb clusters =  3 distance threshold =  3.5  , nb feuilles =  312  runtime =  1.9  ms 
# aggloomeratif_cluster_distance("3-spiral.arff",10,0.5)

##
#   shapes
##
# linkage =  single  nb clusters =  4 distance threshold =  0.5  , nb feuilles =  1000  runtime =  5.88  ms 
#
# linkage =  complete  nb clusters =  4 distance threshold =  2.5  , nb feuilles =  1000  runtime =  11.14  ms 
# linkage =  complete  nb clusters =  4 distance threshold =  3.0  , nb feuilles =  1000  runtime =  11.15  ms 
#
# linkage =  average  nb clusters =  4 distance threshold =  1.5  , nb feuilles =  1000  runtime =  10.89  ms 
# linkage =  average  nb clusters =  4 distance threshold =  2.0  , nb feuilles =  1000  runtime =  10.83  ms 
# aggloomeratif_cluster_distance("shapes.arff",10,0.5)

##
#   fourty
##
#aggloomeratif_cluster_distance("fourty.arff",0.5,0.01)

####################################################################################

#  Clustering hiérarchique en utilisant un nombre de clusters                      #
    
####################################################################################

def aggloomeratif_cluster_k(name,k_max):
    
    
    data = arff.loadarff ( open ( name, 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
    datanp = np.array(datanp)
    f0 = datanp[: , 0] # tous les elements de la premiere colonne
    f1 = datanp[: , 1] # tous les elements de la deuxieme colonne
    link=['single','ward','complete','average']
    # Donnees dans datanp
    linked_mat = shc.linkage ( datanp , 'single')
    plt.figure ( figsize = ( 12 , 12 ) )
    shc.dendrogram ( linked_mat,orientation = 'top' , distance_sort = 'descending' , show_leaf_counts = False )
    plt.title ( " Dendogramme " )
    plt.show ()
    
    
    for l in link :
        print()
        for k in np.arange(2,k_max+1,1):
            # set distance_threshold ( 0 ensures we compute the full tree )
            tps1 = time.time ()
            model = cluster.AgglomerativeClustering ( linkage = l, n_clusters = k )
            model = model.fit ( datanp )
            tps2 = time.time ()
            labels = model.labels_
            #kres = model.n_clusters_
            leaves = model.n_leaves_
        
            # Affichage clustering
            plt.scatter ( f0 , f1 , c = labels , s = 8 )
            plt.title (  f" nb clusters = {k}, linkage = {l}" )
            plt.show ()
            print (" linkage = " ,l ," nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
            


### Test : aggloomeratif_cluster()

##
#   smile1
##
#  linkage =  single  nb clusters =  4  , nb feuilles =  1000  runtime =  5.46  ms 
#aggloomeratif_cluster_k("smile1.arff",6)


##
#   donut3
##
# linkage =  single  nb clusters =  3  , nb feuilles =  999  runtime =  6.86  ms 
#aggloomeratif_cluster_k("donut3.arff",5)


##
#   3-spiral
##
# linkage =  single  nb clusters =  3  , nb feuilles =  312  runtime =  1.21  ms 
#aggloomeratif_cluster_k("3-spiral.arff",4)


##
#   shapes
##
# linkage =  single  nb clusters =  4  , nb feuilles =  1000  runtime =  5.55  ms 
#
# linkage =  ward  nb clusters =  4  , nb feuilles =  1000  runtime =  12.61  ms 
#
# linkage =  complete  nb clusters =  4  , nb feuilles =  1000  runtime =  10.65  ms 
#
# linkage =  average  nb clusters =  4  , nb feuilles =  1000  runtime =  10.6  ms 
#aggloomeratif_cluster_k("shapes.arff",5)

##
#   fourty
##
#aggloomeratif_cluster_k("fourty.arff",45)



########################################################################################################

#  Métrique d'évaluations : Clustering hiérarchique en utilisant un nombre de clusters  et la limite sur le seuil de distance   #
    
########################################################################################################

def aggloomeratif_cluster_k_m(name,k_max):
    
    
    data = arff.loadarff ( open ( name, 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
    datanp = np.array(datanp)
    f0 = datanp[: , 0] # tous les elements de la premiere colonne
    f1 = datanp[: , 1] # tous les elements de la deuxieme colonne
    link=['single','ward','complete','average']
    # Donnees dans datanp
    linked_mat = shc.linkage ( datanp , 'single')
    plt.figure ( figsize = ( 12 , 12 ) )
    shc.dendrogram ( linked_mat,orientation = 'top' , distance_sort = 'descending' , show_leaf_counts = False )
    plt.title ( " Dendogramme " )
    plt.show ()
    k_tab=[]
    silhouette_tab=[]
    
    for l in link :
        print()
        for k in np.arange(2,k_max+1,1):
            # set distance_threshold ( 0 ensures we compute the full tree )
            tps1 = time.time ()
            model = cluster.AgglomerativeClustering ( linkage = l, n_clusters = k )
            model = model.fit ( datanp )
            tps2 = time.time ()
            labels = model.labels_
            #kres = model.n_clusters_
            leaves = model.n_leaves_
        
            # Affichage clustering
            plt.scatter ( f0 , f1 , c = labels , s = 8 )
            plt.title (  f" nb clusters = {k}, linkage = {l}" )
            plt.show ()
            print (" linkage = " ,l ," nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
            
            silhouette_avg = silhouette_score(datanp, labels)
            print ( "linkage = " ,l , ", nb clusters = " ,k ,"silhouette_score = ", round(silhouette_avg,5), " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
            
            k_tab.append(k)
            silhouette_tab.append(silhouette_avg)
            
        plt.plot(k_tab,silhouette_tab)
        plt.xlabel('k')
        plt.ylabel('silhouette score')
        plt.title(f"link {l} : Variation of silhouette score when k increase")
        plt.show()
        k_tab=[]
        silhouette_tab=[]
 
def aggloomeratif_cluster_distance_m(name,d_max,d_step):
    
    data = arff.loadarff ( open ( name, 'r') )
    datanp = [ [ x [ 0 ] ,x [ 1 ] ] for x in data [ 0 ] ]
    datanp = np.array(datanp)
    f0 = datanp[: , 0] # tous les elements de la premiere colonne
    f1 = datanp[: , 1] # tous les elements de la deuxieme colonne
    link=['single','ward','complete','average']
    # # Donnees dans datanp
    # linked_mat = shc.linkage ( datanp , 'single')
    # plt.figure ( figsize = ( 12 , 12 ) )
    # shc.dendrogram ( linked_mat,orientation = 'top' , distance_sort = 'descending' , show_leaf_counts = False )
    # plt.title ( " Dendogramme " )
    # plt.show ()
    d_tab=[]
    silhouette_tab=[]
    
    for l in link :
        cluster_f = True
        print()
        for d in np.arange(0.01,d_max,d_step):
            if cluster_f :
                # set distance_threshold ( 0 ensures we compute the full tree )
                tps1 = time.time ()
                model = cluster.AgglomerativeClustering ( distance_threshold = d , linkage = l , n_clusters = None )
                model = model.fit ( datanp )
                tps2 = time.time ()
                labels = model.labels_
                k = model.n_clusters_
                #leaves = model.n_leaves_
                    
                # Affichage clustering
                plt.scatter ( f0 , f1 , c = labels , s = 8 )
                plt.title ( f" linkage = {l} : nb clusters = {k},  distance_threshold = {d}" )
                plt.show ()
                    
               
                if k == 1 : 
                    cluster_f = False
                else : 
                    silhouette_avg = silhouette_score(datanp, labels)
                    print ( "linkage = " ,l , ", nb clusters = " ,k , ", distance threshold = ",d ,"silhouette_score = ", round(silhouette_avg,5), " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )
                    d_tab.append(d)
                    silhouette_tab.append(silhouette_avg)
                    
        plt.plot(d_tab,silhouette_tab, marker='+',color='b' )
        plt.xlabel('distance threshold')
        plt.ylabel('silhouette score')
        plt.title(f"linkage {l} : Variation of silhouette score when k increase")
        plt.show()
        d_tab=[]
        silhouette_tab=[]
 
    
# silhouette_avg = silhouette_score(datanp, labels)
# print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)

###
#    Distance
###           
#aggloomeratif_cluster_distance_m("donut3.arff", 0.1,0.01)


###
#    k
###    
#aggloomeratif_cluster_k_m("donut3.arff",5)
