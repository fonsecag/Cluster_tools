

parameters={	
	'n_cores':8, # negative numbers means number of total cores minus 

	'load_dataset':{
		'post_processing':None,
		'post_processing_args':[],


		'var_funcs':{
			0:'func:r_to_dist',
			1:'func:extract_R_concat',
			2:'func:extract_F_concat',
			3:'func:extract_E',
			4:'func:f_to_dist',
		}, #end of 'var_funcs'
	},


	'clusters':{
		'save_npy' : True,
		'init_scheme':[0, 1],
		#indices below define the clustering scheme
		0:{
		    'type':'func:agglomerative_clustering', #string, types: Agglomerative, Kmeans
		    'n_clusters':10,
		    'initial_number':10000,
		    'distance_matrix_function':'func:distance_matrix_euclidean',
		    'linkage':'complete',
		    'cluster_choice_criterion':'func:smallest_max_distance_euclidean',
		    
		    'var_index':0,
		    },
		1:{
		    'type':'func:agglomerative_clustering', #string, types: Agglomerative, Kmeans
		    'n_clusters':5,
		    'initial_number':10000,
		    'distance_matrix_function':'func:distance_matrix_euclidean',
		    'linkage':'complete',
		    'cluster_choice_criterion':'func:smallest_max_distance_euclidean',
		    
		    'var_index':4,
		    },
		2:{
		    'type':'func:agglomerative_clustering', #string, types: Agglomerative, Kmeans
		    'n_clusters':200,
		    'initial_number':10000,
		    'distance_matrix_function':'func:distance_matrix_euclidean',
		    'linkage':'complete',
		    'cluster_choice_criterion':'func:smallest_max_distance_euclidean',
		    
		    'var_index':4,
		    },
	}, #end of 'clusters'
}


# functions

