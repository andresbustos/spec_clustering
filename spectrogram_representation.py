#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:01:26 2020

Script that encodes images and then applies clustering techniques.

@author: u5501
"""

import os
from networks import *
import glob
import cv2
import numpy as np
import sklearn.cluster
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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


class spectrograms():
	"""
	Class to deal witht the spectrograms. Needs a 'spec_folder' with the 
	spectrograms in png format.
	
	Performs clustering with several methods and writes output data.
	"""
	def __init__(self, outfolder, specs_folder, heat_type_file, pca_comp=-1):
		
		self.outfolder=outfolder
		self.specs_folder = specs_folder
		self.specs = glob.glob(os.path.join(self.specs_folder, '*png'))
        
		if not os.path.exists(outfolder):
			os.mkdir(outfolder)			
		
		#load the encoder
		img_input, levels = get_vgg_encoder(input_height=257,  input_width=368)    #(7, 7, 512)
		#img_input, levels = get_resnet50_encoder() #(7, 7, 2048)
		output = levels[-1]
		model  = Model(img_input, output)
		model.output_width = model.output_shape[2]
		model.output_height = model.output_shape[1]
		model.model_name = "spectrogram_encoder"
		self.model=model
		print(model.summary())

		# loop over the spectrograms and calculate representation
		self.Xraw = []
		self.shot_numbers = []#to save data in a dataframe
		for i, f in enumerate(self.specs):
			if (i %25==0):
				print('Encoding spectrograms: ', float(i)/len(self.specs)*100, ' %') 
			img = np.array([cv2.imread(f)])[0]/255.
			img = cv2.resize( img, (model.input_shape[2], model.input_shape[1]), interpolation=cv2.INTER_NEAREST)
			out = model.predict_on_batch([[img]])[0]
			self.Xraw.append(out.flatten())
			self.shot_numbers.append(f[-9:-4])

		
		if (pca_comp > 1):
			pca = PCA(n_components=pca_comp)
			pca.fit(self.Xraw)
			self.X = pca.transform(self.Xraw)
		else:
			self.X = self.Xraw

		print('\nDimensions of the model input: ', model.input_shape[2]* model.input_shape[1]*model.input_shape[3])
		print('Dimensions of the image encoding: ', self.Xraw[0].shape)		

		
		self.X_embedded = TSNE(n_components=2).fit_transform(self.X)  #2D embedding
		
		self.num_clusters = [1,2,3,4,5,6,7,8,10,12,16,20,24,28,32,36,40]#,44,48,52,56,60,64,68,72,76,80,100,120]	

		#To make multiplots in a w*w grid
		self.w = 5

		#for future plots except for graphs, which have another names
		self.shape_dict = {'ECH+both injectors':'D', 'ECH+NBI1':'^',
				'ECH+NBI2':'v','Both NBI start-up':'o',
				'NBI1 start-up':'*','NBI2 start-up':'+', 
				'No NBI plasma. No AE':'s'}

		df = pd.read_csv(heat_type_file, index_col='shot_WDIA')
		self.type = []
		for s in self.shot_numbers:
			s = int(s)
			self.type.append(df.loc[s]['NBI scenario 2'])

		plt.figure(dpi=200)
		plt.subplots_adjust(left = 0.1, right = 0.95, bottom = 0.35, top = 0.9, wspace = -0.1, hspace=0.2)
		plt.title('Heating mode histogram')
		plt.ylabel('number of discharges')
		df['NBI scenario 2'].hist(xrot=45, bins=7, alpha=0.75, width=0.7)
		plt.savefig(os.path.join(self.outfolder, 'heating_hist.png'))
		plt.clf()
		plt.cla()
		
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

	def spec_kmeans(self, outfolder, sel_nc=4):
		"""
		Kemans clustering over the data self.X. For each number of clusters
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
			clusters = alg.predict(self.X)
			inertias.append(alg.inertia_)
			df['Nc='+str(nc)] = clusters
			
			dest_folder =  os.path.join(outfolder, 'Nc_'+str(nc).zfill(2)+'_clusters' )
			if not os.path.exists(dest_folder):
				os.mkdir(dest_folder)

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
#		kmeans   = sklearn.cluster.KMeans(n_clusters=sel_nc, precompute_distances=True).fit(self.X)
#		clusters = kmeans.predict(self.X)
#		self.distribute_pictures(outfolder, sel_nc, clusters)
#

	def spec_kmedoids(self, outfolder, sel_nc=4):
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
			inertias.append(alg.inertia_)
			df['Nc='+str(nc)] = clusters
			
			dest_folder =  os.path.join(outfolder, 'Nc_'+str(nc).zfill(2)+'_clusters' )
			if not os.path.exists(dest_folder):
				os.mkdir(dest_folder)
			
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
			
#			fig = plt.figure(dpi=200)
#			plt.scatter(self.X_embedded[:,0], self.X_embedded[:,1],c=clusters, cmap=plt.get_cmap('rainbow'), alpha=0.65)
#			plt.title('Nc = ' + str(nc))
#			plt.savefig(os.path.join(dest_folder,'tSNE_Kmedoids_'+str(nc).zfill(3)+'.png'))
#			plt.clf()
			
			
			#for each cluster, make a summary plot of w*w spectrograms
			shapes  = [self.shape_dict[self.type[i]] for i in range(len(self.X))]
			cmap = plt.get_cmap('rainbow')
			norm = Normalize(vmin=0, vmax=max(clusters))
			fig = plt.figure(dpi=200)
			for i,x in enumerate(self.X_embedded):
				plt.scatter(x[0], x[1],  c=[cmap(norm(clusters[i]))], alpha=0.65, marker=shapes[i])
			plt.title('Nc = ' + str(nc))
			plt.savefig(os.path.join(dest_folder,'tSNE_Kmedoids_'+str(nc).zfill(3)+'.png'))
			plt.clf()		
		
		self.save_shape_dict(outfolder, self.shape_dict)

		self.plot_elbow(inertias, outfolder)
	
		#distribute spectrograms in sepparate folders according to cluster membership
#		kmeans   = sklearn.cluster.KMeans(n_clusters=sel_nc, precompute_distances=True).fit(self.X)
#		clusters = kmeans.predict(self.X)
#		self.distribute_pictures(outfolder, sel_nc, clusters)
	def plot_elbow(self, inertias, outfolder):
		"""
			Elbow method plot
		"""
		fig = plt.figure(dpi=200)
		plt.plot(self.num_clusters, inertias, 'o-')	
		plt.xlabel('number of clusters')
		plt.ylabel('inertia')
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

	def spec_agglomerative(self, outfolder, sel_nc=4):
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
#			clusters = alg.predict(self.X)
			df['Nc='+str(nc)] = clusters
			
			dest_folder =  os.path.join(outfolder, 'Nc_'+str(nc).zfill(2)+'_clusters' )
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
#			
#			fig = plt.figure(dpi=200)
#			plt.scatter(self.X_embedded[:,0], self.X_embedded[:,1],c=clusters, cmap=plt.get_cmap('rainbow'), alpha=0.65)
#			plt.title('Nc = ' + str(nc))
#			plt.savefig(os.path.join(dest_folder,'tSNE_Agglomerative_'+str(nc).zfill(3)+'.png'))
#			plt.clf()
			
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
		For each number of clusters	we plot the corresponding tSNE with the 
		points colored according to	cluster membership.

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
			shapes  = [self.shape_dict[self.type[i]] for i in range(len(self.X))]

			fig = plt.figure(dpi=200)
			for i,x in enumerate(self.X_embedded):
				plt.scatter(x[0], x[1],  c=[cmap(norm(clusters[i]))], alpha=0.65, marker=shapes[i])
			plt.title('Nc = ' + str(nc))
			plt.savefig(os.path.join(dest_folder,'tSNE_DBSCAN_'+str(nc).zfill(3)+'.png'))
			plt.clf()		
			plt.cla()
			fig=None
			
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

#####################################################################

#  Input parameters and execution.

#####################################################################
              
# Here we set the output folder name, the folder with the input images,
# the file with the heating type information and the number of clusters we
# wanna use.

# Takes some time to initialize the image encodings.
		
outfolder           = 'results_temp'
specs_folder        = 'data'
heat_type_file      = '20200630_list_5000_with_NBI_scenario.csv'	
specs               = spectrograms(outfolder, specs_folder, heat_type_file)
specs.num_clusters  = [2,3,4,5] #,6,7,8,10,12,16,20,24,28,32,36,40,44,48,52,56,60,64]	


# Now we do the work, clustering and/or applying PCA

specs.spec_kmeans('KMeans_noPCA')
# specs.spec_kmedoids('KMedoids_noPCA')
# specs.spec_agglomerative('Agglomerative_noPCA')
# specs.spec_DBSCAN('DBSCAN_noPCA', epsilon=0.72857) #len(specs.Xraw[0])/1000)
# np.savetxt(os.path.join(outfolder, "spectrograms_encoded_RAW.csv"), specs.Xraw, delimiter=",")


# specs.make_similarity_graph('Graph_cosine_noPCA')
# specs.make_similarity_graph('Graph_euclidean_noPCA', metric='euclidean')

# for n_pca in[8,16,32,64,128]:
#	specs.apply_pca(n_pca)
#	specs.spec_kmeans('KMeans_PCA'+str(n_pca))
#	specs.spec_kmedoids('KMedoids_PCA'+str(n_pca))
# 	specs.spec_agglomerative('Agglomerative_PCA'+str(n_pca))
# 	specs.spec_DBSCAN('DBSCAN_PCA'+str(n_pca), epsilon=n_pca*3)  #tentative
#	specs.make_similarity_graph('Graph_cosine_PCA'+str(n_pca))
#	specs.make_similarity_graph('Graph_euclidean_PCA'+str(n_pca), metric='euclidean')
#	np.savetxt(os.path.join(outfolder, 'spectrograms_encoded_PCA_'+str(n_pca).zfill(3)+'.csv'), specs.X, delimiter=",")


# 
# for n_pca in[4,8,16,32,64,128,256,512]:
# 	specs.apply_pca(n_pca)
