

parameters={	
	'n_cores':8, # negative numbers means number of total cores minus 

	'load_dataset':{
		'post_processing':None,
		'post_processing_args':[],


		'var_funcs':{
			0:'func:r_to_dist',
			1:'func:extract_R',
			2:'func:extract_F_concat',
			3:'func:extract_E',
		}, #end of 'var_funcs'
	},

	'train_models':{
		'load_func':'func:load_mbGDML_model', # needs to also load trianing set
		'n_mers':3,
		'model_info_func':None,

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
				'predict_func':'func:mbgdml_predict_F',
				'batch_size':100,
				'input_var_index':1,
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
				#replace default paras here
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

	'error_graph_default':{
		'matplotlib_style':'seaborn',

		'reverse_order':False,  #if False, orders from lowest to highest (left to right), reversed otherwise
		'order_by_error':True,
		'label_cluster_index':True,
		'cluster_index_label_fontsize':6,

		# AXES
		'x_axis_label':'Cluster index',
		'y_axis_label':'Prediction error',
		'axis_label_fontsize':12,

		# HOR LINE
		'horizontal_cluster_average':False,
		'horizontal_line':True,
		'horizontal_line_linewidth':1,
		'horizontal_label_fontsize':10,
		'horizontal_line_color':'black',
		'horizontal_label_text':'Mean',

		# POPULATION
		'include_population':True,
		'population_fontsize':10,
		'population_color':'blue',
		'population_linewidth':1,
		'population_alpha':1,
		'population_text':'',

		# BAR VISUALS
		'bar_color':'cluster_gradient', # error_gradient
		'bar_color_gradient':[(.2,.8,.2),(.8,.8,.2),(.8,.2,.2)],
		'bar_width':.93,


		# MISC
		'title_fontsize':18,
		'figsize':(9, 6),
		'transparent_background':False,
		'DPI':300,

		# 'x_axis_label':"Cluster index",
		# 'y_axis_label':r'Force prediction error ($?$)',
		# 'axis_label_fontsize':10,
		# 'axis_tick_size':10,
		# 'axis_linewidth':1.3,

		# 'bar_width':.95,
		# 'bar_color':'error_gradient', #cluster_gradient
		# 'bar_color_gradient':[(.2,.8,.2),(.8,.8,.2),(.8,.2,.2)],  #,(.5,.2,.1)

		# 'horizontal_line':True, #if True, includes a horizontal line to show the average error 
		# 'horizontal_line_linewidth':1,
		# 'horizontal_line_color':(.2,.2,.2), 
		# 'horizontal_label_fontsize':9,
		# 'horizontal_label_text':'overall error',


		# 'fig_size':(5,5), #fig size
		# 'transparent_background':False ,

		# 'include_population':True,  #indicates the cluster population for every cluster 
		# 'population_color':'blue',
		# 'population_alpha':1,
		# 'population_linewidth':1,
		# 'population_fontsize':10,
	},

	'fine_clustering':{
		'fine_indices_func':'func:cluster_above_mse',
		'fine_indices_func_args':[1.1],
		'clustering_scheme':[0],

		'indices_func':'func:within_cluster_weighted_err_N', 
		'indices_func_args':[],
	},

	'R_to_xyz':{
		'var_index':1,
		# needs to be in the form of R_concat 
		# (so shape = (n_samples, n_dim*n_atoms))

		'z':'func:z_from_npz_dataset',
		# is given self and dataset by default
	},


}


# functions

