#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:01:26 2020

Script that encodes images and then applies clustering techniques.

@author: u5501
"""

import os
from networks import *
import glob2
import cv2
import numpy as np
import sklearn.cluster
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colorbar as cbar
import pandas as pd
from sklearn_extra.cluster import KMedoids
from numpy.linalg import norm
import igraph as ig
import matplotlib
from shutil import copyfile
import sys
from sklearn.cluster import AgglomerativeClustering
import random
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import contingency_matrix
from tempfile import TemporaryFile
import pickle
from som import SOM
from timeit import default_timer as timer
from math import floor
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_array, check_consistent_length
import warnings
from sklearn.utils.multiclass import type_of_target
from datetime import datetime
import json

start = datetime.now()
start_time = start.strftime("%H:%M:%S")

class spectrograms():
    """
    Class to deal with the spectrograms. Needs a 'spec_folder' with the 
    spectrograms in png format.
    
    Performs clustering with several methods and writes output data.
    """
    def __init__(self, outfolder, specs_folder, heat_type_file, par, pca_comp=-1, svd_comp=-1, search=False):
        
        self.outfolder    = outfolder
        self.specs_folder = specs_folder
        self.specs        = glob2.glob(os.path.join(self.specs_folder, '*png'))
        self.params       = par
        
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)            
        
        #load the encoder
        img_input, levels   = get_vgg_encoder(input_height=257,  input_width=368) 
        output              = levels[-1]
        model               = Model(img_input, output)
        model.output_width  = model.output_shape[2]
        model.output_height = model.output_shape[1]
        model.model_name    = "spectrogram_encoder"
        self.model          = model
        print(model.summary())

        # loop over the spectrograms and calculate representation
        self.Xraw         = []
        self.shot_numbers = []#to save data in a dataframe
        
        (model, self.Xraw, self.shot_numbers) = self.compute_or_load_VGG(model, par['action'], filename, self.specs)
        if (search):
            with open(os.path.join('reductions','SVD'+filename+'_'+str(svd_comp)+'.pkl'),'rb') as f: # load saved specs
                svd = pickle.load(f)            
#            with open(os.path.join('reductions','Vt'+filename+'_20.pkl'),'rb') as f: # load saved specs
#                Vt20 = pickle.load(f)
            self.X  = svd.transform(self.Xraw)
            return
                
#        for i, f in enumerate(self.specs):
#            if (i %25==0):
#                print('Encoding spectrograms: ', float(i)/len(self.specs)*100, ' %') 
#            img = np.array([cv2.imread(f)])[0]/255.
#            img = cv2.resize( img, (model.input_shape[2], model.input_shape[1]), interpolation=cv2.INTER_NEAREST)
#            out = model.predict_on_batch([[img]])[0]
#            self.Xraw.append(out.flatten())
#            self.shot_numbers.append(f[-9:-4])


        # apply or not PCA
        if (pca_comp > 1):
            if (par['actionPCA'] == 'compute'):
                pca    = PCA(n_components=pca_comp)
                pca.fit(self.Xraw)
                self.X = pca.transform(self.Xraw)
                with open(os.path.join('reductions',filename+str(len(self.specs))+'_'+str(pca_comp)+'PCA.pkl'), 'wb') as f:  #save variables
                    pickle.dump(self.X, f)
            else:
                print('...Using saved data for PCA...')
                with open(os.path.join('reductions',filename+str(len(self.specs))+'_'+str(pca_comp)+'PCA.pkl'),'rb') as f: # load saved specs
                    self.X = pickle.load(f)
        # apply or not SVD
        elif (svd_comp > 1):
            if (par['actionSVD'] == 'compute'):
                self.X=self.svd_decomposition(self.Xraw)
                with open(os.path.join('reductions',filename+str(len(self.specs))+'_'+str(svd_comp)+'SVD.pkl'), 'wb') as f:  #save variables
                    pickle.dump(self.X, f)
            else:
                print('...Using saved data for SVD...')
                with open(os.path.join('reductions',filename+str(len(self.specs))+'_'+str(svd_comp)+'SVD.pkl'),'rb') as f: # load saved specs
                    self.X = pickle.load(f)
        else:
            self.X = self.Xraw

        print('\nDimensions of the model input: ', model.input_shape[2]* model.input_shape[1]*model.input_shape[3])
        print('Dimensions of the image encoding: ', self.Xraw[0].shape)        
        

        self.X_embedded   = TSNE(n_components=2).fit_transform(self.X)  #2D embedding
#        self.num_clusters = [1,2,3,4,6,8,10,12,16,20,24,28,32,36,40]

        #To make multiplots in a w*w grid
        self.w = 5

       
        #for future plots except for graphs, which have another names
#        self.shape_dict = {'ECH followed by both  injectors. Longer NBI2':'D', 'ECH followed by NBI1':'^',
#                'ECH followed by NBI2':'v','ECH followed by both  injectors. Longer NBI1':'o',
#                'ECH followed by both  injectors. Same length':'*','NBI1 start-up followed by NBI2':'+', 
#                'NBI1 start-up':'s', 'both NBI start-up. Longer NBI2': '1', 'both NBI start-up. Longer NBI1' : 'x', 
#                'NBI2 start-up' : 'p', 'NBI2 start-up followed by NBI1' : 'h', 'NBI1 start-up followed by ECH':'H',
#                'NBI2 start-up followed by ECH':'d','NBI2 start-up followed by NBI1 and ECH':'X'}        
        self.shape_dict = {'ECH followed by both  injectors. Longer NBI2':'o', 'ECH followed by NBI1':'o',
                'ECH followed by NBI2':'o','ECH followed by both  injectors. Longer NBI1':'o',
                'ECH followed by both  injectors. Same length':'o','NBI1 start-up followed by NBI2':'o', 
                'NBI1 start-up':'o', 'both NBI start-up. Longer NBI2': 'o', 'both NBI start-up. Longer NBI1' : 'o', 
                'NBI2 start-up' : 'o', 'NBI2 start-up followed by NBI1' : 'o', 'NBI1 start-up followed by ECH':'o',
                'NBI2 start-up followed by ECH':'o','NBI2 start-up followed by NBI1 and ECH':'o'}        
        self.type = []         
         
        df = pd.read_csv(par['heat_type_file'], index_col='shot_WDIA')

        for s in self.shot_numbers:
            s = int(float(s))
            self.type.append(df.loc[s]['NBI scenario'])
       
        '''
        plt.figure(dpi=200)
        plt.subplots_adjust(left = 0.1, right = 0.95, bottom = 0.35, top = 0.9, wspace = -0.1, hspace=0.2)
        plt.title('Heating mode histogram')
        plt.ylabel('number of discharges')
        #df['NBI scenario 2'].hist(xrot=45, bins=7, alpha=0.75, width=0.7)
        plt.savefig(os.path.join(self.outfolder, 'heating_hist.png'))
        plt.clf()
        plt.cla()
        '''
        
    def apply_pca(self, npca):
        pca = PCA(n_components=npca)
        pca.fit(self.Xraw)
        self.X = pca.transform(self.Xraw)
        
    def save_shape_dict(self, folder, d):
        f = open(os.path.join(folder, 'shape_legend.dat'), 'wt')
        f.write('Heating\tShape\n')
        for k in d.keys():
            f.write(k +'\t' + d[k]+'\n')
        f.close()

    def spec_kmeans(self, outfolder, n_it, sel_nc=4):
        """
        Kmeans clustering over the data self.X. For each number of clusters
        we plot the corresponding tSNE with the points colored according to
        cluster membership.
        
        We also plot the inertias versus the number of clusters, for the
        elbow method, and save data in a csv file.
        """            
            
        outfolder = os.path.join(self.outfolder,outfolder)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)

        inertias     = []    
        df           = pd.DataFrame()
        df['shot']   = self.shot_numbers
        for nc in self.num_clusters:
            print('Clustering in ' + str(nc) + ' clusters')
            alg      = sklearn.cluster.KMeans(n_clusters=nc, precompute_distances=True).fit(self.X)
            self.kmeans = alg
            clusters = alg.predict(self.X)
            self.clusters = clusters
            clusters = self.clean_cluster(clusters,nc)
            inertias.append(alg.inertia_)
            df['Nc='+str(nc)] = clusters
            
            dest_folder =  os.path.join(outfolder, 'Nc_'+str(nc).zfill(2)+'_clusters'+str(n_it) )
            if not os.path.exists(dest_folder):
                os.mkdir(dest_folder)

            np.savez(os.path.join(dest_folder, self.params['saving_clusters']), clusters)
            self.make_clustering_histogram(clusters, os.path.join(dest_folder,'cluster_size.png'))
            df['cluster'] = clusters
            df.sort_values(by='shot',inplace=True)                        
            df.to_csv(os.path.join(dest_folder,'membership.csv'), index=None)

            cmap = plt.get_cmap('rainbow')
            norm = Normalize(vmin=0, vmax=max(clusters))

            for ic in range(nc):
                sel_specs = [a for a in np.where(clusters==ic)[0]]
                n_el      = len(sel_specs)
                num_pics  =  min(len(sel_specs),self.w**2)
                sel_specs = random.sample(sel_specs, num_pics)
                filenames  = sorted([self.specs[i] for i in sel_specs])
            
                fig=plt.figure(dpi=200)
                plt.title('Cluster ' + str(ic) + ' - ' + str(n_el) + ' elements '  +' - Total_Clusters='+ str(nc), size=8, pad=12, color=[cmap(norm(ic))][0], fontweight='bold' )
                plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.9, wspace = -0.1, hspace=0.2)
                plt.axis('off')
                for i in range(1, num_pics + 1):
                    fig.add_subplot(self.w, self.w, i)
                    plt.rc('font', size=6)   
                    plt.axis('off')
                    img =  np.array([cv2.imread(filenames[i-1])])[0]
                    img = cv2.resize( img, (self.model.input_shape[2], self.model.input_shape[1]), interpolation=cv2.INTER_NEAREST)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img, cmap='ocean')
                    plt.title('#'+os.path.basename(filenames[i-1])[-9:-4], pad=1)
    
                outpath = os.path.join(os.path.join(dest_folder, 'cluster_'+str(ic).zfill(3)+'_overview.png') )
                plt.savefig(outpath)
                plt.clf()
                fig=None
            
            #for each cluster, make a summary plot of w*w spectrograms
            shapes  = [self.shape_dict[self.type[i]] for i in range(len(self.X))]
            fig = plt.figure(dpi=200)
            for i,x in enumerate(self.X_embedded):
                plt.scatter(x[0], x[1],  c=[cmap(norm(clusters[i]))], alpha=0.65, marker=shapes[i])
            plt.title('Nc = ' + str(nc))
            plt.savefig(os.path.join(dest_folder,'tSNE_Kmeans_'+str(nc).zfill(3)+'.png'))
            plt.clf()    
            fig = None
            
        self.save_shape_dict(outfolder, self.shape_dict)
        
        
        self.plot_elbow(inertias, outfolder)
    
        #save dataframe to file
        #distribute spectrograms in sepparate folders according to cluster membership
#        kmeans   = sklearn.cluster.KMeans(n_clusters=sel_nc, precompute_distances=True).fit(self.X)
#        clusters = kmeans.predict(self.X)
#        self.distribute_pictures(outfolder, sel_nc, clusters)
#
    def spec_som(self, outfolder, n_it, sel_nc=4):
        '''
        SOM clustering over the data self.X. 
        '''
        outfolder = os.path.join(self.outfolder,outfolder)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        inertias = []
        for nc in self.num_clusters:
            # Build a ncx1 SOM 
            print('Clustering in ' + str(nc) + ' clusters')
            som = SOM(m=1, n=nc, dim=self.X[0].shape[0])
            
            # Fit it to the data
            som.fit(self.X, epochs=50, initiate=self.params['initialitation'])
            self.som=som
            inertias.append(som._inertia_)
            
            # Assign each datapoint to its predicted cluster
            clusters = som.predict(self.X)
            self.clusters = clusters
            clusters = self.clean_cluster(clusters,nc)
            
            df           = pd.DataFrame()
            df['shot']   = self.shot_numbers
            df['Nc='+str(nc)] = clusters

            dest_folder =  os.path.join(outfolder, 'Nc_'+str(nc).zfill(2)+'_clusters'+str(n_it) )
            if not os.path.exists(dest_folder):
                os.mkdir(dest_folder)
            
            np.savez(os.path.join(dest_folder, self.params['saving_clusters']), clusters)
            np.savez(os.path.join(dest_folder, self.params['saving_weights']), self.som.weights)
            self.make_clustering_histogram(clusters, os.path.join(dest_folder,'cluster_size.png'))
            df['cluster'] = clusters
            df.sort_values(by='shot',inplace=True)                        
            df.to_csv(os.path.join(dest_folder,'membership.csv'), index=None)
            
            cmap = plt.get_cmap('rainbow')
            norm = Normalize(vmin=0, vmax=max(clusters))
        
            for ic in range(nc):
                sel_specs = [a for a in np.where(clusters==ic)[0]]
                n_el      = len(sel_specs)
                num_pics  =  min(len(sel_specs),self.w**2)
                sel_specs = random.sample(sel_specs, num_pics)
                filenames  = sorted([self.specs[i] for i in sel_specs])
            
                fig=plt.figure(dpi=200)
                plt.title('Cluster ' + str(ic) + ' - ' + str(n_el) + ' elements '  +' - Total_Clusters='+ str(nc), size=8, pad=12, color=[cmap(norm(ic))][0], fontweight='bold' )
                plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.9, wspace = -0.1, hspace=0.2)
                plt.axis('off')
                for i in range(1, num_pics + 1):
                    fig.add_subplot(self.w, self.w, i)
                    plt.rc('font', size=6)   
                    plt.axis('off')
                    img =  np.array([cv2.imread(filenames[i-1])])[0]
                    img = cv2.resize( img, (self.model.input_shape[2], self.model.input_shape[1]), interpolation=cv2.INTER_NEAREST)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img, cmap='ocean')
                    plt.title('#'+os.path.basename(filenames[i-1])[-9:-4], pad=1)
    
                outpath = os.path.join(os.path.join(dest_folder, 'cluster_'+str(ic).zfill(3)+'_overview.png') )
                plt.savefig(outpath)
                plt.clf()
                fig=None
            plt.close('all') #to close all opened figures during the execution
#            self.heating_method_connection(clusters, nc)
            
            shapes  = [self.shape_dict[self.type[i]] for i in range(len(self.X))]
            fig = plt.figure(dpi=200)
            for i,x in enumerate(self.X_embedded):
                plt.scatter(x[0], x[1],  c=[cmap(norm(clusters[i]))], alpha=0.65, marker=shapes[i])
            plt.title('Nc = ' + str(nc),fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.savefig(os.path.join(dest_folder,'tSNE_SOM_'+str(nc).zfill(3)+'.png'))
            plt.clf()    
            fig = None
        
        self.save_shape_dict(outfolder, self.shape_dict)
        self.plot_elbow(inertias, outfolder)
        
    def spec_kmedoids(self, outfolder, n_it, sel_nc=4):
        """
        Kmedoidss clustering over the data self.X. For each number of clusters
        we plot the corresponding tSNE with the points colored according to
        cluster membership.
        
        We also plot the inertias versus the number of clusters, for the
        elbow method, and save data i na csv file.
        """            

        outfolder = os.path.join(self.outfolder,outfolder)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        
        inertias     = []    
        df           = pd.DataFrame()
        df['shot']   = self.shot_numbers
        for nc in self.num_clusters:
            print('Clustering in ' + str(nc) + ' clusters')
            alg      = KMedoids(n_clusters=nc).fit(self.X)
            clusters = alg.predict(self.X)
            self.clusters = clusters
            clusters = self.clean_cluster(clusters,nc)
            inertias.append(alg.inertia_)
            df['Nc='+str(nc)] = clusters
            
            dest_folder =  os.path.join(outfolder, 'Nc_'+str(nc).zfill(2)+'_clusters'+str(n_it) )
            if not os.path.exists(dest_folder):
                os.mkdir(dest_folder)
                
            np.savez(os.path.join(dest_folder, self.params['saving_clusters']), clusters)
            self.make_clustering_histogram(clusters, os.path.join(dest_folder,'cluster_size.png'))
            df = pd.DataFrame(columns=['shot','cluster'])
            df['cluster'] = clusters
            df.sort_values(by='shot',inplace=True)                                                
            df.to_csv(os.path.join(dest_folder,'membership.csv'), index=None)
            
            #for each cluster, make a summary plot of w*w spectrograms
            cmap = plt.get_cmap('rainbow')
            norm = Normalize(vmin=0, vmax=max(clusters))
            for ic in range(nc):
                sel_specs = [a for a in np.where(clusters==ic)[0]]
                n_el      = len(sel_specs)
                num_pics  =  min(len(sel_specs),self.w**2)
                sel_specs = random.sample(sel_specs, num_pics)
                filenames  = sorted([self.specs[i] for i in sel_specs])
            
                fig=plt.figure(dpi=200)
                plt.title('Cluster ' + str(ic) + ' - ' + str(n_el) + ' elements '  +' - Total_Clusters='+ str(nc), size=8, pad=12 , color=[cmap(norm(ic))][0], fontweight='bold' )
                plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.9, wspace = -0.1, hspace=0.2)
                plt.axis('off')
                for i in range(1, num_pics + 1):
                    fig.add_subplot(self.w, self.w, i)
                    plt.rc('font', size=6)   
                    plt.axis('off')
                    img =  np.array([cv2.imread(filenames[i-1])])[0]
                    img = cv2.resize( img, (self.model.input_shape[2], self.model.input_shape[1]), interpolation=cv2.INTER_NEAREST)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img, cmap='ocean')
                    plt.title('#'+os.path.basename(filenames[i-1])[-9:-4], pad=1)
    
                outpath = os.path.join(os.path.join(dest_folder, 'cluster_'+str(ic).zfill(3)+'_overview.png'))
                plt.savefig(outpath)
                plt.clf()
                plt.cla()
                fig=None
            
#            fig = plt.figure(dpi=200)
#            plt.scatter(self.X_embedded[:,0], self.X_embedded[:,1],c=clusters, cmap=plt.get_cmap('rainbow'), alpha=0.65)
#            plt.title('Nc = ' + str(nc))
#            plt.savefig(os.path.join(dest_folder,'tSNE_Kmedoids_'+str(nc).zfill(3)+'.png'))
#            plt.clf()
            
            
            #for each cluster, make a summary plot of w*w spectrograms
#            shapes  = [self.shape_dict[self.type[i]] for i in range(len(self.X))]
#            cmap = plt.get_cmap('rainbow')
#            norm = Normalize(vmin=0, vmax=max(clusters))
#            fig = plt.figure(dpi=200)
#            for i,x in enumerate(self.X_embedded):
#                plt.scatter(x[0], x[1],  c=[cmap(norm(clusters[i]))], alpha=0.65, marker=shapes[i])
#            plt.title('Nc = ' + str(nc))
#            plt.savefig(os.path.join(dest_folder,'tSNE_Kmedoids_'+str(nc).zfill(3)+'.png'))
#            plt.clf()        
#        self.save_shape_dict(outfolder, self.shape_dict)

        self.plot_elbow(inertias, outfolder)
    
        #distribute spectrograms in sepparate folders according to cluster membership
#        kmeans   = sklearn.cluster.KMeans(n_clusters=sel_nc, precompute_distances=True).fit(self.X)
#        clusters = kmeans.predict(self.X)
#        self.distribute_pictures(outfolder, sel_nc, clusters)

    def plot_elbow(self, inertias, outfolder):
        """
            Elbow method plot
        """
        fig = plt.figure(dpi=200)
        plt.plot(self.num_clusters, inertias, 'o-')    
        plt.xlabel('Número de clústeres',fontsize=15)
        plt.ylabel('Inercia', fontsize=15)
        plt.savefig(os.path.join(outfolder, 'elbow_method.png'))
        plt.clf()
        fig = None
        
    def distribute_pictures(self, outfolder, sel_nc, clusters):
        """
        Distributes the spectrograms (self.specs) in 'sel_nc' folders corresponding
        to the partition defined in 'clusters'. Writes all in 'outfolder'.
        """
        for c in range(sel_nc):
            d = os.path.join(os.path.join(outfolder,'cluster_'+str(c).zfill(3) ))
            try:
                os.mkdir(d)
            except:
                pass    
        for i,spec in enumerate(self.specs):
            c = clusters[i]
            dest  = os.path.join(os.path.join(outfolder,'cluster_'+str(c).zfill(3) ), os.path.basename(spec))
            copyfile(spec, dest)

        #for each cluster, make a summary plot of w*w spectrograms
        for ic in range(sel_nc):
            sel_specs = [a for a in np.where(clusters==ic)[0]]
            random.shuffle(sel_specs)
            num_pics  =  min(len(sel_specs),self.w**2)
        
            fig=plt.figure(dpi=500)
            plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.9, wspace = -0.1, hspace=0.2)
            for i in range(1, num_pics + 1):
                img = np.random.randint(10, size=(self.w,self.w))
                fig.add_subplot(w, w, i)
                plt.rc('font', size=6)   
                plt.axis('off')
                filename = self.specs[sel_specs[i-1]]
                img = np.array([cv2.imread(filename)])[0]
                img = cv2.resize( img, (self.model.input_shape[2], self.model.input_shape[1]), interpolation=cv2.INTER_NEAREST)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img, cmap='ocean')
                plt.title('#'+os.path.basename(filename)[-9:-4], pad=1)

            outpath = os.path.join(os.path.join(outfolder,'cluster_'+str(ic).zfill(3) ), 'cluster_'+str(ic).zfill(3)+'_overview.png')
            plt.savefig(outpath)
            plt.clf()
            fig=None

    def spec_agglomerative(self, outfolder, n_it, sel_nc=4):
        """
        Agglomerative clustering over the data self.X. For each number of clusters
        we plot the corresponding tSNE with the points colored according to
        cluster membership.

        """            
        outfolder = os.path.join(self.outfolder,outfolder)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        
        inertias     = []    
        df           = pd.DataFrame()
        df['shot']   = self.shot_numbers
        for nc in self.num_clusters:
            print('Clustering in ' + str(nc) + ' clusters')
            clusters      = AgglomerativeClustering(n_clusters=nc).fit_predict(self.X)
            self.clusters = clusters
            clusters = self.clean_cluster(clusters,nc)
#            clusters = alg.predict(self.X)
            df['Nc='+str(nc)] = clusters
            
            dest_folder =  os.path.join(outfolder, 'Nc_'+str(nc).zfill(2)+'_clusters'+str(n_it) )
            if not os.path.exists(dest_folder):
                os.mkdir(dest_folder)

            self.make_clustering_histogram(clusters, os.path.join(dest_folder,'cluster_size.png'))
            np.savez(os.path.join(dest_folder, self.params['saving_clusters']), clusters)
            df['cluster'] = clusters
            df.sort_values(by='shot',inplace=True)                                        
            df.to_csv(os.path.join(dest_folder,'membership.csv'), index=None)
            
            #for each cluster, make a summary plot of w*w spectrograms
            cmap = plt.get_cmap('rainbow')
            norm = Normalize(vmin=0, vmax=max(clusters))
            for ic in range(nc):
                sel_specs = [a for a in np.where(clusters==ic)[0]]
                n_el      = len(sel_specs)
                num_pics  =  min(len(sel_specs),self.w**2)
                sel_specs = random.sample(sel_specs, num_pics)
                filenames  = sorted([self.specs[i] for i in sel_specs])
            
                fig=plt.figure(dpi=200)
                plt.title('Cluster ' + str(ic) + ' - ' + str(n_el) + ' elements '  +' - Total_Clusters='+ str(nc), size=8, pad=12, color=[cmap(norm(ic))][0], fontweight='bold'  )
                plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.9, wspace = -0.1, hspace=0.2)
                plt.axis('off')
                for i in range(1, num_pics + 1):
                    fig.add_subplot(self.w, self.w, i)
                    plt.rc('font', size=6)   
                    plt.axis('off')
                    img =  np.array([cv2.imread(filenames[i-1])])[0]
                    img = cv2.resize( img, (self.model.input_shape[2], self.model.input_shape[1]), interpolation=cv2.INTER_NEAREST)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img, cmap='ocean')
                    plt.title('#'+os.path.basename(filenames[i-1])[-9:-4], pad=1)
    
                outpath = os.path.join(os.path.join(dest_folder, 'cluster_'+str(ic).zfill(3)+'_overview.png'))
                plt.savefig(outpath)
                plt.clf()    
                fig=None
            
#            fig = plt.figure(dpi=200)
#            plt.scatter(self.X_embedded[:,0], self.X_embedded[:,1],c=clusters, cmap=plt.get_cmap('rainbow'), alpha=0.65)
#            plt.title('Nc = ' + str(nc))
#            plt.savefig(os.path.join(dest_folder,'tSNE_Agglomerative_'+str(nc).zfill(3)+'.png'))
#            plt.clf()
            
            #for each cluster, make a summary plot of w*w spectrograms
            shapes  = [self.shape_dict[self.type[i]] for i in range(len(self.X))]
            cmap = plt.get_cmap('rainbow')
            norm = Normalize(vmin=0, vmax=max(clusters))
            fig = plt.figure(dpi=200)
            for i,x in enumerate(self.X_embedded):
                plt.scatter(x[0], x[1],  c=[cmap(norm(clusters[i]))], alpha=0.65, marker=shapes[i])
            plt.title('Nc = ' + str(nc))
            plt.savefig(os.path.join(dest_folder,'tSNE_Agglomerative_'+str(nc).zfill(3)+'.png'))
            plt.clf()        
            plt.cla()
            fig=None
        self.save_shape_dict(outfolder, self.shape_dict)



    def spec_DBSCAN(self, outfolder, epsilon):
        """
        Density-Based Spatial Clustering of Applications with Noise over the data self.X.
        For each number of clusters    we plot the corresponding tSNE with the 
        points colored according to    cluster membership.

        """            
    
        outfolder = os.path.join(self.outfolder,outfolder)    
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        
        df           = pd.DataFrame()
        df['shot']   = self.shot_numbers
        clusters     = sklearn.cluster.DBSCAN(min_samples=5, eps=epsilon, p=2).fit_predict(self.X)
        nc = max(clusters)+1
        df['Nc='+str(nc)] = clusters
        
        dest_folder =  outfolder
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)

        self.make_clustering_histogram(clusters, os.path.join(dest_folder,'cluster_size.png'))
        df['cluster'] = clusters
        df.sort_values(by='shot',inplace=True)                                        
        df.to_csv(os.path.join(dest_folder,'membership.csv'), index=None)        
            
        #for each cluster, make a summary plot of w*w spectrograms
        cmap = plt.get_cmap('rainbow')
        norm = Normalize(vmin=0, vmax=max(clusters))
        for ic in range(nc):
            sel_specs = [a for a in np.where(clusters==ic)[0]]
            n_el      = len(sel_specs)
            num_pics  =  min(len(sel_specs),self.w**2)
            sel_specs = random.sample(sel_specs, num_pics)
            filenames  = sorted([self.specs[i] for i in sel_specs])
        
            fig=plt.figure(dpi=200)
            plt.title('Cluster ' + str(ic) + ' - ' + str(n_el) + ' elements '  +' - Total_Clusters='+ str(nc), size=8, pad=12 , color=[cmap(norm(ic))][0], fontweight='bold'  )
            plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.9, wspace = -0.1, hspace=0.2)
            plt.axis('off')
            for i in range(1, num_pics + 1):
                fig.add_subplot(self.w, self.w, i)
                plt.rc('font', size=6)   
                plt.axis('off')
                img =  np.array([cv2.imread(filenames[i-1])])[0]
                img = cv2.resize( img, (self.model.input_shape[2], self.model.input_shape[1]), interpolation=cv2.INTER_NEAREST)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img, cmap='ocean')
                plt.title('#'+os.path.basename(filenames[i-1])[-9:-4], pad=1)

            outpath = os.path.join(os.path.join(dest_folder, 'cluster_'+str(ic).zfill(3)+'_overview.png'))
            plt.savefig(outpath)
            plt.clf()
            plt.cla()
            fig=None
            
            
            #for each cluster, make a summary plot of w*w spectrograms
            '''
            shapes  = [self.shape_dict[self.type[i]] for i in range(len(self.X))]

            fig = plt.figure(dpi=200)
            for i,x in enumerate(self.X_embedded):
                plt.scatter(x[0], x[1],  c=[cmap(norm(clusters[i]))], alpha=0.65, marker=shapes[i])
            plt.title('Nc = ' + str(nc))
            plt.savefig(os.path.join(dest_folder,'tSNE_DBSCAN_'+str(nc).zfill(3)+'.png'))
            plt.clf()        
            plt.cla()
            fig=None
            '''
        self.save_shape_dict(outfolder, self.shape_dict)



    def make_similarity_graph(self, outfolder, metric='cosine',factor_edges = 10):
        """
          Graph clustering
          metric = 'cosine' or 'euclidean'
          factor edges: L = factor_edges*N    
        """        
    
        outfolder = os.path.join(self.outfolder,outfolder)
        try:
            os.mkdir(outfolder)
        except:
            pass
        
        # CREATE GRAPH
        N     = len(self.X)
        nodes = [ a for a in range(N)]
        links = []
        w_list = []
        #loop over all pair of nodes, calculate possible link weight
        for n1 in nodes:
            norm1 = norm(self.X[n1])
            for n2 in nodes[n1+1:]:
                if (metric=='cosine'):
                    w = np.dot(self.X[n1],self.X[n2])/norm1/norm(self.X[n2]) #cosine similarity
                elif (metric=='euclidean'):
                    d = norm(self.X[n1]-self.X[n2]) #euclidean distance
                    if d<0.01:
                        w = 100.
                    else:
                        w = 1./d        
                else:
                    print('make_similarit_graph: metric not implemented')
                    sys.exit()
                
                links.append([n1,n2])        
                w_list.append(w)
        w_list = [ a / max(w_list) for a in w_list]  #some normalization
        
        #dataframe with all possible links
        df_l = pd.DataFrame(links, columns=['src','tgt'])
        df_l['w'] = w_list
        df_l.sort_values(by='w', ascending=False, inplace=True)
        L = min(len(df_l), N*factor_edges)
        
        #Now create the graph with the L strongest links
        g= ig.Graph()
        g.add_vertices(nodes)
        #g.vs['size'] = [abs(a)*6 +6 for a in mode_size]
        g.add_edges( df_l[0:L][['src','tgt']].values)
        g.es['weight'] = df_l[0:L]['w'].values
        
        #calculate comunities and assign a color to each community
        comms = g.community_multilevel(weights=g.es['weight'])
        self.make_clustering_histogram(np.array(comms.membership), os.path.join(outfolder, 'community_size.png'))

        num_comms    = 12
        sel_comms = sorted(range(len(comms.sizes())), key = lambda sub: comms.sizes()[sub])[-num_comms:][::-1] 
        all_colors  = [ b for b in matplotlib.colors.TABLEAU_COLORS.values()]
        all_colors += [ b for b in matplotlib.colors.BASE_COLORS.values()]
        my_colors   = {}
        cont        = 0
        for s in sel_comms:
            my_colors.update({s:all_colors[cont]})
            cont += 1
                
        # write data to csv files
        f = open(os.path.join(outfolder, 'graph_links.csv') , 'wt')
        f.write('Source,Target,weight\n')    
        for n1,n2,w in df_l[0:L].values:
            f.write(str(int(n1)) + ',' +str(int(n2))+','+str(w)+'\n' )
        f.close()
            
        f = open(os.path.join(outfolder,'graph_nodes.csv'),'wt')
        f.write('Id,Label,comm\n')
        for i,s in enumerate(self.specs):
            if comms.membership[i] in sel_comms:
                c = comms.membership[i]
            else:
                c  = max(sel_comms) +1 
            f.write(str(i)+','+'#'+s[-9:-4] +',' + str(c)+'\n')
        f.close()

        #for future plots
        shape_dict = {'ECH+both injectors':'diamond', 'ECH+NBI1':'triangle-up',
                'ECH+NBI2':'triangle-down','Both NBI start-up':'circle',
                'NBI1 start-up':'circle','NBI2 start-up':'circle', 
                'No NBI plasma. No AE':'square'}
        
        # print some stuff on screen and make a layout plot
        g.vs['comm'] = comms.membership
        f_readme = open(os.path.join(outfolder,'README.txt'),'wt')
        f_readme.write('  Graph size: ' + str(len(g.vs)) +' ' + str(len(g.es))+'\n')
        f_readme.write('  modularity: ' + str(comms.modularity)+'\n')
        lay= None
        lay = g.layout_fruchterman_reingold(weights=g.es['weight'], niter=5000)#, start_temp=500)
        #lay = g.layout_kamada_kawai(maxiter=150000, kkconst=1000.) 
        #lay = g.layout_lgl(maxiter=10000, repulserad=500)
        #lay= g.layout_graphopt(niter=5000, node_charge=0.5)#, spring_length=10000)
        visual_style = {}
        visual_style["bbox"]         = (800,800)
        visual_style["margin"]       = 100
        visual_style["edge_curved"]  = True
        visual_style['vertex_color'] = [ my_colors[comms.membership[i]] if (comms.membership[i] in my_colors.keys())  else '#000000'  for i in range(len(g.vs))]
        visual_style['layout']       = lay
        g.vs['shape']                = [shape_dict[self.type[i]] for i in nodes]
        ig.plot(comms, mark_groups = False, target= os.path.join(outfolder,'graph_comms.png'), **visual_style)
        
        #distribute spectrograms in sepparate folders according to community membership
        mem_folders = {}
        for c in sel_comms:
            d = os.path.join(outfolder,'comm_'+str(c).zfill(3) )
            mem_folders.update({c:d})
            try:
                os.mkdir(d)
            except:
                pass
            f_readme.write('Community ' + str(c) + ' has ' + str(comms.sizes()[c]) +' elements\n')
                    
        for i,spec in enumerate(self.specs):
            if g.vs['comm'][i] in sel_comms:
                dest  = os.path.join(mem_folders[g.vs['comm'][i]], os.path.basename(spec))
                copyfile(spec, dest)
    
    
        #for each community, make a summary plot of w*w spectrograms
        for ic in sel_comms:
            sel_specs = [a for a in np.where(np.array(comms.membership)==ic)[0]]
            n_el      = len(sel_specs)
            num_pics  =  min(len(sel_specs),self.w**2)
            sel_specs = random.sample(sel_specs, num_pics)
            filenames  = sorted([self.specs[i] for i in sel_specs])
        
            fig=plt.figure(dpi=200)
            plt.title('Cluster ' + str(ic) + ' - ' + str(n_el) + ' elements ', size=8, pad=12, color=my_colors[ic], fontweight='bold' )
            plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.05, top = 0.9, wspace = -0.1, hspace=0.2)
            plt.axis('off')
            for i in range(1, num_pics + 1):
                fig.add_subplot(self.w, self.w, i)
                plt.rc('font', size=6)   
                plt.axis('off')
                img =  np.array([cv2.imread(filenames[i-1])])[0]
                img = cv2.resize( img, (self.model.input_shape[2], self.model.input_shape[1]), interpolation=cv2.INTER_NEAREST)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img, cmap='ocean')
                plt.title('#'+os.path.basename(filenames[i-1])[-9:-4], pad=1)
        
            outpath = os.path.join(os.path.join(outfolder, 'community_'+str(ic).zfill(3)+'_overview.png'))
            plt.savefig(outpath)
            plt.clf()    
            plt.cla()
            fig=None
            
        self.save_shape_dict(outfolder, shape_dict)
        f_readme.close()

    def make_clustering_histogram(self, c, path):
        df = pd.DataFrame({'membership':c})
        plt.figure(dpi=200)
        plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.2, top = 0.9, wspace = -0.1, hspace=0.2)
        plt.title('Number of members of each cluster', size=9)
        plt.xlabel('Cluster ID', size=9)
        bins = sorted(set(c))
        bins.append(len(bins))
        plt.xticks(bins, size=8)
        df['membership'].hist(alpha=0.75, width=0.98, bins=bins, align='left')
        plt.savefig(path)
        plt.clf()
        plt.cla()
        fig=None
        
    def svd_decomposition(self,X):
        '''
        Computes SVD decomposition of data X using the number components that 
        gives a reconstruction error of 1e-2
        '''
#        svd_comp = self.params['svd_comp']
#        [U,S,V] = svd(X,full_matrices=False,compute_uv=True)
        with open(os.path.join('reductions','USV'+filename+'_'+str(178)+'.pkl'), 'rb') as f:  
            [U,S,V] = pickle.load(f)
        
        # To plot singular values
#        primer = S[0]
#        plt.plot(S/primer)
#        plt.yscale('log')        
#        plt.xlabel('i', size=12)        
#        plt.ylabel(r'$\sigma_i/\sigma_1$', size=12)
#        plt.xticks(fontsize=12)
#        plt.yticks(fontsize=12)
        
        r=np.linalg.matrix_rank(X)
        n_comp=r
        for i in range(1,r):
            err=np.sqrt(sum(map(lambda x:x*x,S[i:r])))/np.sqrt(sum(map(lambda x:x*x,S[0:i])))
            if (err < 1e-3):
                print("El error para ", i, " componentes es ", err)
                n_comp=i
                self.params["svd_comp"] = n_comp
                break
        svd_decomp = TruncatedSVD(n_components=n_comp)
        svd_decomp.fit(X)
        newX = svd_decomp.transform(X)
        
        with open(os.path.join('reductions','SVD'+filename+'_'+str(2)+'.pkl'), 'wb') as f:  
            pickle.dump(svd_decomp, f) #Saving V to use it to project images
        
        return newX
        
            
    def compute_or_load_VGG(self, model, action, filename, specs):
        '''
        Depending on the action input ('compute' or 'load'), a VGG embedding is
        computed and saved or loaded. The filename for saving or loading is also
        an input, no extension needed.
        '''
        X = []
        shot_numbers = []
        if (action == 'compute'):
            for i, f in enumerate(specs):
                if (i %25==0):
                    print('Encoding spectrograms: ', float(i)/len(specs)*100, ' %') 
                img = np.array([cv2.imread(f)])[0]/255.
                img = cv2.resize( img, (model.input_shape[2], model.input_shape[1]), interpolation=cv2.INTER_NEAREST)
                out = model.predict_on_batch([[img]])[0]
                X.append(out.flatten())
                shot_numbers.append(f[-9:-4])
                
            with open(os.path.join('reductions',filename+str(len(specs))+'.pkl'), 'wb') as f:  #save variables
                pickle.dump([model,X,shot_numbers], f)

        elif (action == 'load'):
            print('...Using saved data...')
            with open(os.path.join('reductions',filename+str(len(specs))+'.pkl'),'rb') as f: # load saved specs
                [model,X,shot_numbers] = pickle.load(f)
            
        return (model, np.array(X), shot_numbers)
            
    def clean_cluster(self, cluster, cluster_size):
        '''
        Function used to transform a clustering to its "canonical form". 
        For example, [3,3,2,3,2] is equivalent to [0,0,1,0,1]
        '''
        order_by_cluster = {}
        new_cluster=np.zeros((len(cluster),), dtype=int)
        order = []
        for i in range(cluster_size):
            order_by_cluster[i]=[] #create a dictionary with every cluster as key
        
        for i in range(len(cluster)):
            order_by_cluster[cluster[i]].append(i) #insert positions of every cluster
            if (cluster[i] not in order):
                order.append(cluster[i]) #order of appearance of clusters
        
        for i in range(len(order)):
            for pos in order_by_cluster[order[i]]:
                new_cluster[pos]=i #create new cluster

        return new_cluster
    
    ### Functions used to compute rand index
    def check_clusterings(self,labels_true, labels_pred):
        """Check that the labels arrays are 1D and of same dimension.
    
        Parameters
        ----------
        labels_true : array-like of shape (n_samples,)
            The true labels.
    
        labels_pred : array-like of shape (n_samples,)
            The predicted labels.
        """
        labels_true = check_array(
            labels_true, ensure_2d=False, ensure_min_samples=0, dtype=None,
        )
    
        labels_pred = check_array(
            labels_pred, ensure_2d=False, ensure_min_samples=0, dtype=None,
        )
    
        type_label = type_of_target(labels_true)
        type_pred = type_of_target(labels_pred)
    
        if 'continuous' in (type_pred, type_label):
            msg = f'Clustering metrics expects discrete values but received' \
                  f' {type_label} values for label, and {type_pred} values ' \
                  f'for target'
            warnings.warn(msg, UserWarning)
    
        # input checks
        if labels_true.ndim != 1:
            raise ValueError(
                "labels_true must be 1D: shape is %r" % (labels_true.shape,))
        if labels_pred.ndim != 1:
            raise ValueError(
                "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
        check_consistent_length(labels_true, labels_pred)
    
        return labels_true, labels_pred

    def pair_confusion_matrix(self, labels_true, labels_pred):
        """Pair confusion matrix arising from two clusterings.
        """
        labels_true, labels_pred = self.check_clusterings(labels_true, labels_pred)
        n_samples = np.int64(labels_true.shape[0])
    
        # Computation using the contingency data
        contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
        n_c = np.ravel(contingency.sum(axis=1))
        n_k = np.ravel(contingency.sum(axis=0))
        sum_squares = (contingency.data ** 2).sum()
        C = np.empty((2, 2), dtype=np.int64)
        C[1, 1] = sum_squares - n_samples
        C[0, 1] = contingency.dot(n_k).sum() - sum_squares
        C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
        C[0, 0] = n_samples ** 2 - C[0, 1] - C[1, 0] - sum_squares
        return C
    
    def rand_index(self, cluster1, cluster2):
        '''
        Computes a similarity measure between two clusterings by considering all 
        pairs of samples and counting pairs that are assigned in the same or 
        different clusters in two given clusterings.
        '''
        contingency = self.pair_confusion_matrix(cluster1, cluster2)
        numerator = contingency.diagonal().sum()
        denominator = contingency.sum()
    
        if numerator == denominator or denominator == 0:
            # Special limit cases: no clustering since the data is not split;
            # or trivial clustering where each document is assigned a unique
            # cluster. These are perfect matches hence return 1.0.
            return 1.0
    
        return round(numerator / denominator,3)
    ### End of functions used to compute rand index

    def compare_saved_clusters(self, path1, path2, n_clusters):
        '''
        This function is used to compare two different clusterings using Rand index.
        These clusterings are saved in .npz files in different folders depending on
        the algorithm used.
        Four versions of every clustering are compared.
        '''
        rands = []
        str_cluster=str(n_clusters)
        for i in range(self.params['n_iteraciones']):
            for j in range(self.params['n_iteraciones']):
                subpath1 = 'Nc_'+str_cluster.zfill(2)+'_clusters'+str(i)
                subpath2 = 'Nc_'+str_cluster.zfill(2)+'_clusters'+str(j)
                clusters1 = np.load(os.path.join(path1, subpath1, self.params['saving_clusters']))
                clusters2 = np.load(os.path.join(path2, subpath2, self.params['saving_clusters']))
                rands.append(self.rand_index(clusters1['arr_0'],clusters2['arr_0']))
        return rands


    def plot_rand_vs_reduc(self,comps,rands,reduc):  
        '''
        This function is used to compare different number of 
        components used to compute PCA or SVD reduction with the rand 
        indeces of the clustering made with that reduction vs no reduction
        '''
        axes = plt.gca()
        axes.set_ylim([0.75,1])
        plt.plot(comps,rands,'-o')
        plt.xlabel('Número de componentes ' + reduc,fontsize=12)
        plt.ylabel('Indice de Rand',fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
    def compare_versions(self, versions_list, n_clusters):
        '''
        We use rand index to compare a list of different clusterings of the 
        same data
        '''
        n_versions = len(versions_list)
        means = np.ones([n_versions,n_versions])
        stdevs = np.ones([n_versions,n_versions])
        for i in range(n_versions):
            for j in range(n_versions):
                 (means[i][j],stdevs[i][j])= self.mean_std(self.compare_saved_clusters(
                         os.path.join(outfolder,versions_list[i]),os.path.join(
                                 outfolder,versions_list[j]),n_clusters))
        self.print_mean_std(means,stdevs,n_versions)
       
    def compare_one_method_versions(self, method_folder, n_clusters):
        '''
        We use rand index to compare a list of different clusterings of the 
        same data
        '''
        n_it=self.params['n_iteraciones']
        rands_array = np.ones(n_it*n_it)
        str_cluster=str(n_clusters)
        for i in range(n_it):
            line = []
            for j in range(n_it):
                subpath1 = 'Nc_'+str_cluster.zfill(2)+'_clusters'+str(i)
                subpath2 = 'Nc_'+str_cluster.zfill(2)+'_clusters'+str(j)
                clusters1 = np.load(os.path.join(outfolder,method_folder,subpath1,self.params['saving_clusters']))
                clusters2 = np.load(os.path.join(outfolder,method_folder,subpath2,self.params['saving_clusters']))
                rands_array[i*n_it+j]=self.rand_index(clusters1['arr_0'],clusters2['arr_0'])
                line.append(rands_array[i*n_it+j])
            print(line)
            
        mean, std = self.mean_std(rands_array)
        print("media: ", mean, "+/-", std)
        
    def print_mean_std(self, means, stdevs, size):
        '''
        Given a matrix of means and a matrix of standard deviations, this 
        function prints the matrix with both values in every component
        ---
        Example:
            means  = [[0.950, 0.97],
                     [0.88,  0.711]]
            stdevs = [[0.12, 0.02],
                     [0.11,  0.005]]
            self.print_mean_std(means, stdevs, 2) = [[0.950(12), 0.97(2)],
                                                    [0.88(11),    0.711(5)]]
        '''
        for i in range(size):
            line = []
            for j in range(size):
                dec=len(str(stdevs[i][j]))
                std=stdevs[i][j]*10**(dec-2)
                line.append(str(format(means[i][j], '.'+str(len(str(stdevs[i][j]))-2)+'f')) + "(" + str(int(std)) + ")")
            print(line)
        
        
    def mean_std(self,rands):
        '''
        Returns a tuple with the mean and standard deviation of a given list of values
        '''
        rands_filter = list(filter(lambda x: (x != 1.0), rands)) 
        
        std = str(float('%.1g' % np.std(rands_filter)))
        if (std[-1]=='1'):
            std = str(float('%.2g' % np.std(rands_filter)))
            
        return (round(np.mean(rands_filter),len(std)-2),std)
    
    ### Functions used to make the heating types graph
    def heating_method_connection(self, clusters, nc, folder):
        '''
        Creates a graphic with the different heating methods used in each
        cluster. It includes a chart with the amount of heating methods per
        cluster
        '''
        data = self.count_heating_types(clusters)
        data = np.array(data).transpose()
        
        columns = np.arange(nc)
        rows = list(self.shape_dict.keys())[::-1]
        
        value_increment = 1000
        
        # Get some pastel shades for the colors
        colors = plt.cm.terrain(np.linspace(0, 0.5, len(rows)))
        n_rows = len(data)
        
        index = np.arange(len(columns)) + 0.3
        bar_width = 0.4
        
        # Initialize the vertical-offset for the stacked bar chart.
        y_offset = np.zeros(len(columns))
        
        # Plot bars and create text labels for the table
        fig, ax = plt.subplots(dpi=150)
        fig.set_size_inches(10, 5)
        ax.get_xaxis().set_visible(False)
        for row in range(n_rows):
            ax.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
            y_offset = y_offset + data[row]
        # Reverse colors and text labels to display the last value at the top.
        colors = colors[::-1]
        
        # Add a table at the bottom of the axes
        ax.table(cellText=data[::-1],
                              rowLabels=rows,
                              rowColours=colors,
                              colLabels=columns,
                              loc='bottom')
        
        # Adjust layout to make room for the table:
        fig.subplots_adjust(left=0.3, bottom=0.4)
        
        plt.ylabel("Heating method quantity".format(value_increment))
        plt.title('Type of heating method per cluster')
        
        plt.show()
        
        plt.savefig(os.path.join(folder, 'heating_method_'+str(nc)+'_clusters.png'))

    def count_heating_types(self, clusters):
        '''
        Counts the amount of spectrograms of every heating method in all 
        clusters
        '''
        cluster_dict = {}
        clusters= self.clusters
        for i in range(len(clusters)):
            if (clusters[i] in cluster_dict):
                cluster_dict[clusters[i]].append(i)
            else:
                cluster_dict[clusters[i]] = [i]                
        heat_types = []
        for key in cluster_dict.keys():
            heat_types_temp = []
            for value in cluster_dict[key]:
                heat_types_temp.append(self.type[value])  
            heat_types.append(self.count_types_in_array(heat_types_temp))
        return heat_types 
                
    def count_types_in_array(self, heat_array):
        '''
        Counts the amount of spectrograms of every heating method in an array
        '''
        count = []
        for heat_method in self.shape_dict.keys():
            count.append(heat_array.count(heat_method))
        return count
    ### End of functions used to make the heating types graph
    
def read_parameters(parname):
    '''
    Used to read parameters stored in a JSON file
    '''
    with open(parname,'rt') as f:
        p=json.load(f)
    return p

#def compare_cluster_sizes(path1,path2);

        

#  Input parameters and execution.

#####################################################################
              
# Here we set the output folder name, the folder with the input images,
# the file with the heating type information and the number of clusters we
# wanna use.

# Takes some time to initialize the image encodings.
            
# Parameters to define the folder name depending on the initialitation and wether
# or not we use PCA or SVD?
par=read_parameters("params.json")

if (par['initialitation'] == 'pca'):
    par['ini_file'] = 'initial'
if (par['pca_comp'] != -1):
    par['pca_file'] = 'PCA'+str(par['pca_comp'])
if (par['svd_comp'] != -1):
    par['svd_file'] = 'SVD'+str(par['svd_comp'])
    
folder              = 'SOM_'+par['pca_file']+par['svd_file']+par['ini_file']
outfolder           = 'results_test_'+par['filename']
specs_folder        = par['filename']

# Clustering execution time begins
total_time = 0
start = timer()
specs               = spectrograms(outfolder, specs_folder, par['heat_type_file'], par, par['pca_comp'], par['svd_comp'])
specs.num_clusters  =[10] 
specs.spec_som(folder,1)
end = timer()
total_time = (end - start)
print("Execution time " + folder + ": " + str(floor(total_time/60)) + " min " + str(floor(total_time%60)) + " s")


'''
#specs.spec_kmeans('KMeans_noPCA')
# specs.spec_kmedoids('KMedoids_noPCA')
# specs.spec_agglomerative('Agglomerative_noPCA')
# specs.spec_DBSCAN('DBSCAN_noPCA', epsilon=0.72857) #len(specs.Xraw[0])/1000)
# np.savetxt(os.path.join(outfolder, "spectrograms_encoded_RAW.csv"), specs.Xraw, delimiter=",")


############## To compute Rand index between two saved clusters ##############

# folder options: KMeans_noPCA, SOM_noPCA, SOM_PCA490, SOM_noPCAinitial, ...
#path1 = outfolder + '/SOM_noPCA' #folder of the first clustering
#path2 = outfolder + '/SOM_noPCA'    #folder of the second clustering
#specs.compare_saved_clusters(path1,path2,specs.num_clusters)
#specs.compare_saved_clusters(path1,path2,specs.num_clusters)

# specs.make_similarity_graph('Graph_cosine_noPCA')
# specs.make_similarity_graph('Graph_euclidean_noPCA', metric='euclidean')

# for n_pca in[8,16,32,64,128]:
#    specs.apply_pca(n_pca)
#    specs.spec_kmeans('KMeans_PCA'+str(n_pca))
#    specs.spec_kmedoids('KMedoids_PCA'+str(n_pca))
#     specs.spec_agglomerative('Agglomerative_PCA'+str(n_pca))
#     specs.spec_DBSCAN('DBSCAN_PCA'+str(n_pca), epsilon=n_pca*3)  #tentative
#    specs.make_similarity_graph('Graph_cosine_PCA'+str(n_pca))
#    specs.make_similarity_graph('Graph_euclidean_PCA'+str(n_pca), metric='euclidean')
#    np.savetxt(os.path.join(outfolder, 'spectrograms_encoded_PCA_'+str(n_pca).zfill(3)+'.csv'), specs.X, delimiter=",")


# 
# for n_pca in[4,8,16,32,64,128,256,512]:
#     specs.apply_pca(n_pca)
'''