# -*- coding: utf-8 -*-
"""
Created on Thu May 13 19:15:48 2021

@author: bsgal
"""

from spectrogram_representation import spectrograms
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os, cv2, json, glob2, pickle
from networks import *

def plot_similars(spec_id,specs_by_distance,search_file):
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
    file = os.path.join(search_file,spec_id+'.png') #esto es para cuando se usa el dataset de espectrogramas
    img =  np.array([cv2.imread(file)])[0]
    img = cv2.resize( img, (specs.model.input_shape[2], specs.model.input_shape[1]), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap='ocean')
    plt.title('SEARCH', pad=1)
              
    spec_num = ''.join(list(filter(lambda x: (x.isdigit()), spec_id)))
    for i in range(1,num_pics+1 if num_pics != 25 else num_pics):
        fig.add_subplot(specs.w, specs.w, i+1)
        plt.rc('font', size=6)   
        plt.axis('off')
#        file = os.path.join(filename,str(specs_by_distance[i-1]).zfill(3)+'.png')
        file = os.path.join(filename,specs_by_distance[i-1]+'.png') # para el dataset de espectrogramas
        img =  np.array([cv2.imread(file)])[0]
        img = cv2.resize( img, (specs.model.input_shape[2], specs.model.input_shape[1]), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img, cmap='ocean')
        plt.title('#'+''.join(list(filter(lambda x: (x.isdigit()),specs_by_distance[i-1]))), pad=1)

    outpath = os.path.join(os.path.join(outfolder, 'similar_specs'+spec_num+'.png') )
    plt.savefig(outpath)
    plt.clf()
    fig=None

def read_parameters(parname):
    '''
    Used to read parameters stored in a JSON file
    '''
    with open(parname,'rt') as f:
        p=json.load(f)
    return p
    
#spec_ids = [44584,39710,33282,29807] # para el dataset de espectrogramas

########## Prepare parameters for execution ########## 
par=read_parameters("params.json")

filename            = par['filename']  
pca_comp            = par['pca_comp']      # number of components of the pca, -1 if no PCA
svd_comp            = par['svd_comp']  #or 178    # number of components of the svd, -1 if no SVD
ini_file            = par['ini_file']
pca_file            = par['pca_file']
svd_file            = par['svd_file']

if (par['initialitation'] == 'pca'):
    ini_file = 'initial'
if (pca_comp != -1):
    pca_file = 'PCA'+str(pca_comp)
if (svd_comp != -1):
    svd_file = 'SVD'+str(svd_comp)
    
folder              = 'SOM_'+pca_file+svd_file+ini_file
outfolder           = 'results_test_'+filename
specs_folder        = filename

########## Compute clustering ##########  

if (os.path.isfile(os.path.join('reductions','specs_'+filename+'_'+svd_file+'.pkl'))):
    with open(os.path.join('reductions','specs_'+filename+'_'+str(svd_file)+'.pkl'),'rb') as f: 
        specs = pickle.load(f)
else:
    print("Can't find file with spectrograms object, it will be created and the clustering will be made ¿?")
    specs               = spectrograms(outfolder, specs_folder, par['heat_type_file'], par, pca_comp,svd_comp)
    specs.num_clusters  =[10] 
    specs.spec_som(folder,1) 
    with open(os.path.join('reductions','specs_'+filename+'_'+svd_file+'.pkl'), 'wb') as f:
        pickle.dump(specs, f)

path_specs=copy(specs.specs)
img_specs=list(map(lambda x: x.split('\\')[1].split('.')[0],path_specs)) 

path_search = glob2.glob(os.path.join(par["path_search"], '*png'))
img_search=list(map(lambda x: x.split('\\')[1].split('.')[0],path_search))

if (img_search == []):
    raise Exception("The search folder is empty, introduce at least one image")

# To compute images embedding and reduction if needed
par['action'] = 'compute'
specs_new = spectrograms(outfolder, par["path_search"], par['heat_type_file'], par, pca_comp,svd_comp,True)

### Search image cluster and order images in that cluster by distance
for spec in img_search:
    if (spec not in img_specs):
        position = img_search.index(spec)
        img = specs_new.X[img_search.index(spec)]
    else:
        position = img_specs.index(spec)
        img = specs.X[position]
        
    cluster = specs.som.predict([img])[0]

    # Elements in the same cluster as spec_id
    cluster_elements = []
    for i in range(len(specs.clusters)):
        if (specs.clusters[i]==cluster):
            cluster_elements.append(img_specs[i])
            
    
    #Compute distance to images to order by similarity 
    distance_dict = {}
    for i in cluster_elements:
        distance_dict[i] = np.linalg.norm(img - specs.X[img_specs.index(i)], axis=0)
    
    distance_dict_order=dict(sorted(distance_dict.items(), key=lambda item: item[1]))
    print("Dictionary ordered by distance",distance_dict_order)                        
    specs_by_distance = list(distance_dict_order.keys())
    
    plot_similars(spec,specs_by_distance,par["path_search"])
    
plt.close('all')
