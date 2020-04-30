from util import*
import util
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from functools import partial 
import multiprocessing as mp 


def smallest_max_distance_euclidean(sample,clusters):
	'''
	Finds the cluster that the given sample belongs to. Simple euclidean distance is used.
	(The metric used should be the same as for the agglomerative clustering.)
	
	Paramters:
		-sample: 
			numpy array containing positions of atoms of one samples
			Dimensions: (n_atoms,n_dimensions)
		
		-clusters: 
			numpy array containing positions within each cluster
			Dimensions: (n_clusters,n_atoms,n_dimensions)
					
	Returns:
		-index of cluster that the sample belongs to / closest cluster
										 
	'''

	g=np.zeros(len(clusters))
	for i in range(len(clusters)):
		g[i]=np.max(np.sum(np.square(clusters[i]-sample),1))   #numpy difference=>clusters[c]-sample elementwise for each c
	return np.argmin(g)
	
def distance_matrix_euclidean(data):

	return euclidean_distances(data,data)

def agglomerative_clustering(self,indices,scheme_index):

	print_ongoing_process("Preparing agglomerative clustering")

	data,N,ind_cluster,cluster_data=None,None,[],[]

	##get n_clusters
	n_clusters_para=self.call_para('clusters',scheme_index,'n_clusters')
	n_clusters=n_clusters_para


	##generate data
	var_index=self.call_para('clusters',scheme_index,'var_index')
	if not (indices is None):
		data=self.vars[var_index][indices]
	else:
		data=self.vars[var_index]


	##get number of initial clusters
	initial_number=self.call_para('clusters',scheme_index,'initial_number')
	if initial_number>1:
		N=initial_number
	else:
		N=len(data)*initial_number


	##prepare agglomerative varsß
	ind_all=np.arange(len(data))
	ind_init=np.random.permutation(ind_all)[:N] 
	data_init=data[ind_init]
	ind_rest=np.delete(ind_all,ind_init)
	data_rest=data[ind_rest]

	print_ongoing_process("Preparing agglomerative clustering",True)

	print_ongoing_process("Computing distance matrix")
	M=self.call_para('clusters',scheme_index,'distance_matrix_function',args=[data_init])
	print_ongoing_process("Computing distance matrix",True)

	##agglo
	print_ongoing_process("Computing tree")
	linkage=self.call_para('clusters',scheme_index,'linkage')

	cinit_labels=AgglomerativeClustering(
		affinity="precomputed",n_clusters=n_clusters,
		linkage=linkage,
	).fit_predict(M)

	print_ongoing_process("Compute tree",True)

	cluster_ind=[]
	for i in range(n_clusters):
		ind=np.concatenate(np.argwhere(cinit_labels==i))

		#convert back to initial set of indices
		ind=ind_init[ind]

		cluster_ind.append(ind.tolist())
		cluster_data.append(np.array(data[cluster_ind[i]]))



	#divide rest into clusters
	#using para['cluster_choice_criterion']
	#+ni to find the old index back from entire dataset
	#print("Clustering rest of data...")`
	#outs=np.trunc(np.linspace(0,len(data_rest),99))

	l_data_rest=len(data_rest)

	if l_data_rest>0:
		for i in range(l_data_rest):
			c=self.call_para('clusters',scheme_index,"cluster_choice_criterion",
				args=[data_rest[i],cluster_data],
			)
			cluster_ind[c].append(ind_rest[i])
			print_percent("Sorting rest of data",i,l_data_rest)

		print_percent("Sorting rest of data",l_data_rest,l_data_rest,True)



	if indices is None:
		return cluster_ind

	#if needed, change the indices of every cluster back corresponding to original data set
	for cl in cluster_ind:
		for i in range(len(cl)):
			cl[i]=indices[cl[i]]

	return cluster_ind

def kmeans_clustering(data_base,indices,clustering_index):
	print_ongoing_process("Preparing KMeans")
	para=data_base.para['clusters'][clustering_index]
	data,n_clusters,cluster_ind=None,para["n_clusters"],[]
	if not (indices is None):
		data=data_base.vars[para["var_index"]][indices]
	else:
		data=data_base.vars[para["var_index"]]

	print_ongoing_process("Preparing KMeans",True)

	print_ongoing_process("Proceed to clustering")
	cluster_labels=MiniBatchKMeans(n_clusters=n_clusters,init="k-means++").fit_predict(data)

	for i in range(n_clusters):
		ind=np.concatenate(np.argwhere(cluster_labels==i).tolist())
		#convert back to initial set of indices
		#no need here
		cluster_ind.append(ind)

	print_ongoing_process("Completed KMeans clustering", True)
	if indices is None:
		return cluster_ind

	#if needed, change the indices of every cluster back corresponding to original data set
	for cl in cluster_ind:
		for i in range(len(cl)):
			cl[i]=indices[cl[i]]

	return cluster_ind

def cluster_do(self,scheme, init_ind = None):

	n_clusters=len(scheme)
	if n_clusters==0:
		return 


	print_cluster_scheme(scheme,self.get_para('clusters'))
	#perform first clusterisation
	index=scheme[0]
 	
	print_cluster_step(1,len(scheme))
	cl_ind=self.call_para('clusters',index,'type',args=[self,init_ind,index])


	#perform further clusterisations  
	for i in range(1,len(scheme)):
		print_cluster_step(i+1,len(scheme))
		index=scheme[i]
		cl_ind_new=[]
		for cl in cl_ind:
			cl_cl_ind=self.call_para('clusters',index,'type',args=[self,cl,index])
			for j in cl_cl_ind:
				cl_ind_new.append(j)


		cl_ind=cl_ind_new

	return cl_ind

def worst_N_clusters(self,N,*args):
	mse=self.cluster_err
	cl_ind=self.init_cluster_indices
	sorted_ind=np.argsort(mse)
	clusters=np.array(cl_ind)[sorted_ind[-N:]]
	ind=np.concatenate(clusters)
	return ind

def cluster_above_mse(self, cl_ind, mse, fact,*args):

	mmse=np.mean(mse)
	cl_ind_new=np.concatenate(np.argwhere(mse>mmse*fact))
	clusters=np.array(cl_ind)[cl_ind_new]
	ind=np.concatenate(clusters)
	return ind

def weighted_distribution(N, weights):
	weights=np.array(weights)/np.sum(weights)
	a=(weights*N)
	b=a.astype(int)
	c=a-b
	s=np.sum(b)

	for i in range(N-s):
		arg=np.argmax(c)
		c[arg]=0
		b[arg]=b[arg]+1

	return b

def within_cluster_weighted_err_N(self, cl_ind, err, N):
	new_ind=[]

	#find cluster errors and pops
	mse=np.array([np.mean(err[x]) for x in cl_ind])
	pop=np.array([len(x) for x in cl_ind])

	weights=(mse/np.sum(mse))*(pop/np.sum(pop))
	Ns=weighted_distribution(N, weights)

	for  i in range(len(cl_ind)):
		ind=np.array(cl_ind[i])
		cl_err=err[ind]
		ni=Ns[i]
		argmax=np.argsort(-cl_err)[:ni]
		new_ind.extend(ind[argmax])

	return new_ind

