# -*- coding: utf-8 -*-
"""
Created on Thu May 13 19:15:48 2021

@author: bsgal
"""

from spectrogram_representation import spectrograms
from copy import copy
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import cv2

def plot_similars(spec_id):
    '''
    Funcion para pintar la imagen buscada y las mas cercanas a esta dentro del 
    cluster que se le ha asignado
    '''
    fig=plt.figure(dpi=200)
    cmap = plt.get_cmap('rainbow')
    norm = Normalize(vmin=0, vmax=max(specs.clusters))
    num_pics  =  min(len(specs_by_distance),specs.w**2)
    plt.title('Most similar elements', size=8, pad=12, color=[cmap(norm(5))][0], fontweight='bold' )
    plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.9, wspace = -0.1, hspace=0.2)
    plt.axis('off')
    
    #Pongo primero la imagen que queremos buscar
    fig.add_subplot(specs.w, specs.w, 1)
    plt.rc('font', size=6)   
    plt.axis('off')
    file = filename+'/'+str(int(spec_id)).zfill(3)+'.png'
#    file = filename+'/'+'spectrogram_'+str(spec_id)+'.png' #esto es para cuando se usa el dataset de espectrogramas
    img =  np.array([cv2.imread(file)])[0]
    img = cv2.resize( img, (specs.model.input_shape[2], specs.model.input_shape[1]), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap='ocean')
    plt.title('SEARCH', pad=1)
              
    for i in range(1,num_pics):
        
        fig.add_subplot(specs.w, specs.w, i+1)
        plt.rc('font', size=6)   
        plt.axis('off')
        file = filename+'/'+str(specs_by_distance[i-1]).zfill(3)+'.png'
#        file = filename+'/'+'spectrogram_'+str(specs_by_distance[i-1])+'.png' # para el dataset de espectrogramas
        img =  np.array([cv2.imread(file)])[0]
        img = cv2.resize( img, (specs.model.input_shape[2], specs.model.input_shape[1]), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img, cmap='ocean')
        plt.title('#'+str(specs_by_distance[i-1]), pad=1)

    outpath = os.path.join(os.path.join(outfolder, 'similar_specs'+str(spec_id)+'.png') )
    plt.savefig(outpath)
    plt.clf()
    fig=None


#print("Indica el numero del espectrograma a comparar")

spec_id_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] #int(input())
#spec_id_list = [44584,39710,33282,29807] # para el dataset de espectrogramas

########## Prepare parameters for execution ########## 
filename            = 'dataset_dummy'  
initialitation      = 'random' # 'random' or 'pca'    
pca_comp            = -1      # number of components of the pca, -1 if no PCA
svd_comp            = -1  #or 178    # number of components of the svd, -1 if no SVD
ini_file            = ''
pca_file            = 'noPCA'
svd_file            = 'noSVD'
if (initialitation == 'pca'):
    ini_file = 'initial'
if (pca_comp != -1):
    pca_file = 'PCA'+str(pca_comp)
if (svd_comp != -1):
    svd_file = 'SVD'+str(svd_comp)
folder              = 'SOM_'+pca_file+svd_file+ini_file
outfolder           = 'results_test_'+filename
specs_folder        = filename
heat_type_file      = '20200630_list_5000_with_NBI_scenario.csv'   
saving_clusters     = 'clusters.npz' #file where the clustering will be saved
action = 'load'        #'compute' or 'load' VGG embedding
actionPCA = 'compute'  #'compute' or 'load' PCA reduction (if needed)
actionSVD = 'load'  #'compute' or 'load' SVD reduction (if needed)

########## Compute clustering ########## 
specs               = spectrograms(outfolder, specs_folder, heat_type_file, pca_comp,svd_comp)
specs.num_clusters  =[2] 
specs.spec_som(folder,1)  
for spec_id in spec_id_list: #para cada imagen, calcular el cluster al que pertenece y pintar las imagenes mas cercanas
    print(spec_id)
    path_specs=copy(specs.specs)
    img_specs=list(map(lambda x: int(x.split('\\')[1].split('.')[0]),path_specs))
#    img_specs=list(map(lambda x: int(x.split('_')[1].split('.')[0]),path_specs)) # para el dataset de espectrogramas


    if (spec_id not in img_specs):
        raise Exception("El espectrograma no se encuentra en la base de datos")
    
    position = img_specs.index(spec_id)
    cluster = specs.som.predict([specs.X[position]])[0]
    
    # Elements in the same cluster as spec_id
    cluster_elements = []
    specs_in_cluster = np.ones
    for i in range(len(specs.clusters)):
        if (specs.clusters[i]==cluster):
            cluster_elements.append(img_specs[i])
            
    
    #Compute distance to images to order by similarity 
    distance_dict = {}
    for i in cluster_elements:
        distance_dict[i] = np.linalg.norm(specs.X[position] - specs.X[img_specs.index(i)], axis=0)
    
    distance_dict_order=dict(sorted(distance_dict.items(), key=lambda item: item[1]))
    print("Dictionary ordered by distance",distance_dict_order)                        
    specs_by_distance = list(distance_dict_order.keys())


    plot_similars(spec_id)
    plt.close('all')  