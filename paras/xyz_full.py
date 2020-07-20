

parameters={	
	'n_cores':-2, # negative numbers means number of total cores minus 

	'load_dataset':{
		'post_processing':None,
		'post_processing_args':[],


		'var_funcs':{
			0:'func:extract_R_concat',
			1:'func:extract_F_concat',
			2:'func:extract_E',
		}, #end of 'var_funcs'
	},


	'clusters':{
		'save_npy' : True,
		'init_scheme':[0],
		#indices below define the clustering scheme
		0:{
			'type':'func:single_cluster', 
			'var_index':0,
			'n_clusters':1,
			},
	}, #end of 'clusters'

	'R_to_xyz':{
		'var_index_R':0,
		'var_index_F':1,
		'var_index_E':2,
		# needs to be in the form of R_concat 
		# (so shape = (n_samples, n_dim*n_atoms))

		'z':'func:z_from_npz_dataset',
		# is given self and dataset by default
	},

}


# functions

