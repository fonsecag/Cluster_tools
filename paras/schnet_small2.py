

parameters={	
	'n_cores':-2, # negative numbers means number of total cores minus 
	'remove_temp_files': False,

	'load_dataset':{
		'schnet_preprocessed_npz':True,

		'var_funcs':{
			0:'func:schnet_r_to_dist',
			1:'func:schnet_extract_F_concat',
			2:'func:schnet_extract_E',
			3:'func:schnet_indices',
		}, #end of 'var_funcs'
	},


	'clusters':{
		'save_npy' : True,
		'init_scheme':[0, 1],
		#indices below define the clustering scheme
		0:{
			'type':'func:kmeans_clustering',
			'n_clusters':10,
			'var_index':0,
			},
		1:{
			'type':'func:kmeans_clustering',
			'n_clusters':5,
			'var_index':2,
		},
		2:{
			'type':'func:agglomerative_clustering', 
			'n_clusters':200,
			'initial_number':20000,
			'distance_matrix_function':'func:distance_matrix_euclidean',
			'linkage':'complete',
			'cluster_choice_criterion':'func:smallest_max_distance_euclidean',
			'var_index':0,
			},
	}, #end of 'clusters'

	'train_models':{
		# 
		'train_func':'func:schnet_train_default',
		'model_ext':'',
		'train_func_args':['para:train_models,schnet_train_args'],

		'load_func':'func:load_schnet_model', # needs to also load trianing set
		'model_info_func':None,
		'schnet_train_args':{
			'batch_size'    : 1,
			'n_features'    : 5, # 64
			'n_gaussians'   : 25,
			'n_interactions': 6,
			'cutoff'        : 5.,
			'learning_rate' : 1e-3,
			'rho_tradeoff'  : 0.1,
			'patience'      : 5,
			'n_epochs'      : 2,
			'n_val'         : 1000
		},
	},


	'predict_error':{
		'error_sub_indices':'func:return_second_argument', #twenty_each
		'main_error_index':0, 
		# the error that is used for those commands that only need one type 
		# of error (like 'train') rather than an entire analysis
		# error will be saved into self.cluster_err (by_cluster must be true!)
		# only main error is calculated if extended = False in calculate_error

		'sample_wise_error_func':'func:MSE_sample_wise',
		'sample_wise_predicts_index':0,
		'sample_wise_comparison_var_index':1,
		# sample-wise only relevant if sample_wise = True in calculate_error

		'predicts':{
			0:{
				'predict_func':'func:schnet_predict_F',
				'input_var_index':3,
			},

		}, # end of predicts


		#RMSE
		0:{
			'predicts_index':0,
			'comparison_var_index':1,			


			'by_cluster':True,
			'error_func':'func:root_mean_squared_error',
			'file_name':'RMSE_graph',
			'save_key':'RMSE_c', #will be saved in save.p under the given key (in 'errors' sub dict)
			'graph':True,
			'graph_paras':{
				'y_axis_label':r'Force prediction error $(kcal/mol\,\AA)$',
				'real_average_key':'RMSE_o'
				#replace default paras here
			},
		},

		#overall RMSE
		1:{
			'predicts_index':0,
			'comparison_var_index':1,			

			'error_func':'func:root_mean_squared_error',
			'save_key':'RMSE_o', #will be saved in save.p under the given key (in 'errors' sub dict)
			'graph':False,
			'graph_paras':{
			},
		},

		#MAE
		2:{
			'predicts_index':0,
			'comparison_var_index':1,			

			'by_cluster':True,
			'error_func':'func:mean_absolute_error',
			'file_name':'MAE_graph',
			'save_key':'MAE_c', #will be saved in save.p under the given key (in 'errors' sub dict)
			'graph':True,
			'graph_paras':{
				'y_axis_label':r'Force prediction error $(kcal/mol\,\AA)$',
				'real_average_key':'MAE_o'
				#replace default paras here
			},
		},

		#overall MAE
		3:{
			'predicts_index':0,
			'comparison_var_index':1,			

			'error_func':'func:mean_absolute_error',
			'save_key':'MAE_o', #will be saved in save.p under the given key (in 'errors' sub dict)
			'graph':False,
			'graph_paras':{
			},
		}, 

		#MSE
		4:{
			'predicts_index':0,
			'comparison_var_index':1,			

			'by_cluster':True,
			'error_func':'func:mean_squared_error',
			'file_name':'MSE_graph',
			'save_key':'MSE_c', #will be saved in save.p under the given key (in 'errors' sub dict)
			'graph':True,
			'graph_paras':{
				'y_axis_label':r'Force prediction error $(kcal/mol\,\AA)^2$',
				'real_average_key':'MSE_o'
			},
		},

		#overall MSE
		5:{
			'predicts_index':0,
			'comparison_var_index':1,			

			'error_func':'func:mean_squared_error',
			'save_key':'MSE_o', #will be saved in save.p under the given key (in 'errors' sub dict)
			'graph':False,
			'graph_paras':{
			},
		},	
	},

	'fine_clustering':{
		'fine_indices_func':'func:cluster_above_mse',
		'fine_indices_func_args':[1.1],
		'clustering_scheme':[2],

		'indices_func':'func:within_cluster_weighted_err_N', 
		'indices_func_args':[],
	}

}


# functions

