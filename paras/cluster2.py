

parameters={	

	'load_dataset':{
		'post_processing':None,
		'post_processing_args':[],


		'var_funcs':{
			0:'func:r_to_minkowski_m0p5',
		}, #end of 'var_funcs'
	},

	'cluster2':{
		'var_index':0,
		'rand_score_threshold': 0.999

		,
		'rand_score_count': 3,
		'n_clusters_min': 2,
		'n_clusters_max': 300,

		'pca':True,
		'pca_kernel': 'rbf',
		'n_components_min': 50,
		'n_components_max':51,
		'n_components_jump':10,
		'rand_n_cl': [14, 20],
	},

}


# functions

