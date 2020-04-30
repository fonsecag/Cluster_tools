

parameters={	

	'clusters':{
		'save_npy' : True,
		'init_scheme':[0],
		#indices below define the clustering scheme
		0:{
		    'type':'func:agglomerative_clustering', #string, types: Agglomerative, Kmeans
		    'n_clusters':2,
		    'initial_number':15000,
		    'distance_matrix_function':'func:distance_matrix_euclidean',
		    'linkage':'complete',
		    'cluster_choice_criterion':'func:smallest_max_distance_euclidean',
		    
		    'var_index':0,
		    },
	}, #end of 'clusters'

	'train_models':{
		'model_ext':'npz',
		'train_func':'func:sgdml_train_default',
		'train_func_args':['para:train_models,sgdml_train_args'],


		'suppress_sgdml_prints':True,
		'load_func':'func:load_sgdml_model', # needs to also load trianing set
		'model_info_func':'func:sgdml_model_info',

		'sgdml_train_args':{
			'n_train':100, # will be handled automatically where needed
						   # for example for improved learning
			'n_test':100,
			'n_valid':100,
			'overwrite':True,
			'command':'all',
			'sigs':None,
			'gdml':False,
			'use_E':False,
			'use_E_cstr':False,
			# 'max_processes':-1, # set manually in the function
			'use_torch':False,
			'solver':'analytic',
			'use_cprsn':False,
		},

	},

	'split_models':{
		'mix_model':True,
		
	},

	'split_inter':{
		'local_error_func':'func:test'
	},

}


# functions

