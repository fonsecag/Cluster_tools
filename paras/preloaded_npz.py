parameters={	

	'train_models':{
		'load_func':'func:load_npz_prepredicted', # needs to also load trianing set
		'model_info_func':None,
	},


	'load_dataset':{
		'post_processing':None, 
		'post_processing_args':[], 

		'schnet_preprocessed_npz':False,

		'var_funcs':{
			0:'func:r_to_dist',
			1:'func:extract_R_concat',
			2:'func:extract_F_concat',
			3:'func:extract_E',
			4:'func:npz_indices'
		}, #end of 'var_funcs'
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
		'sample_wise_comparison_var_index':2,
		# sample-wise only relevant if sample_wise = True in calculate_error

		'predicts':{
			0:{
				'predict_func':'func:npz_prepredicted_F',
				'batch_size':100,
				'input_var_index':4,
			},

		}, # end of predicts


		#RMSE
		0:{
			'predicts_index':0,
			'comparison_var_index':2,			


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
			'comparison_var_index':2,			

			'error_func':'func:root_mean_squared_error',
			'save_key':'RMSE_o', #will be saved in save.p under the given key (in 'errors' sub dict)
			'graph':False,
			'graph_paras':{
			},
		},

		#MAE
		2:{
			'predicts_index':0,
			'comparison_var_index':2,			

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
			'comparison_var_index':2,			

			'error_func':'func:mean_absolute_error',
			'save_key':'MAE_o', #will be saved in save.p under the given key (in 'errors' sub dict)
			'graph':False,
			'graph_paras':{
			},
		}, 

		#MSE
		4:{
			'predicts_index':0,
			'comparison_var_index':2,			

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
			'comparison_var_index':2,			

			'error_func':'func:mean_squared_error',
			'save_key':'MSE_o', #will be saved in save.p under the given key (in 'errors' sub dict)
			'graph':False,
			'graph_paras':{
			},
		},	

	},
}

# functions

