

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
		'init_scheme':[0],
		#indices below define the clustering scheme
		0:{
		    'type':'func:agglomerative_clustering', #string, types: Agglomerative, Kmeans
		    'n_clusters':10,
		    'initial_number':15000,
		    'distance_matrix_function':'func:distance_matrix_euclidean',
		    'linkage':'complete',
		    'cluster_choice_criterion':'func:smallest_max_distance_euclidean',
		    
		    'var_index':4,
		    },
	}, #end of 'clusters'

	'classification':{
		'var_index':0,
		'scaler_func':'func:standard_scaler',  #can be None
		# 'scaler_func':None,

		## RANDOM FOREST -- 0.81 / 1.00
		# 'class_func':'func:random_forest_classifier', 
		# 'n_estimators':100, # 100-200
		# 'max_depth':20, # 20
		# 'min_samples_split':2, # 2 
		# 'criterion':'entropy', # dm
		# 'min_impurity_decrease':0, # 0
		# 'ccp_alpha':0, # 0 


		## EXTREME FOREST -- 0.83 / 1.00
		'class_func':'func:extreme_forest_classifier', 
		'n_estimators':100, # 100
		'max_depth':20, # 20
		'min_samples_split':2, # 2 
		'criterion':'entropy', # dm

		## SVM -- 0.59 / 0.60
		# 'class_func':'func:svm_svc_classifer',
		# 'reg':0.01,

		## Gaussian Process Clf -- 0.71 / 0.85 (also takes years)
		# 'class_func':'func:gaussian_process_classifier', 

		# NN Clf -- 0.83 / 0.99 (relu, adam, constant)
		# 'class_func':'func:neural_network_classifier',
		# 'hidden_layers':(250, 250),
		# 'alpha':.01,
		# 'learning_rate':'constant',
		# 'solver':'adam',
		# 'activation':'relu',

		## AdaBoost Clf -- 0.61 / 0.61
		# 'class_func':'func:AdaBoost_classifier',
		# 'n_estimators':200,
		# 'learning_rate':1,

		'n_points':.7, 
		'perform_test':True,
		'test_func':'func:dotscore_classifier_test',
	},

}


# functions

