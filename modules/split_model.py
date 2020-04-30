from run import MainHandler
from .clustererror import ClusterErrorHandler
from .cluster import ClusterHandler
from .classification import ClassificationHandler
from util import*


class SplitModelHandler(ClusterErrorHandler):

	def __init__(self,args, **kwargs):
		super().__init__(args, **kwargs)
		self.n_stages = self.n_main_stages + 2
		if self.call_para('split_models', 'preload_predict'):
			self.n_stages += 1
		if self.call_para('split_models', 'classify'):
			self.n_stages += 1

	def run_command(self):

		self.model_dir = os.path.join(self.storage_dir, 'models')
		self.model_ext = self.call_para('train_models','model_ext')
		os.mkdir(self.model_dir)

		ClusterHandler.run_command(self)
		# self.cluster_dict = [ 
		# 	{'ind':x, 'rejected':False} for x in self.cluster_indices ]


		## INITIAL MODEL
		self.print_stage('Creating submodels')
		self.create_split_models()

		if self.call_para('split_models', 'classify'):
			self.print_stage('Classification')
			ClassificationHandler.prepare_classifier(self)

		if self.call_para('split_models', 'preload_predict'):
			self.print_stage('Pre-predicting models')
			self.preload_model_predictions()

	def create_split_models(self):
		n_models = len(self.cluster_indices)
		cl_ind = self.cluster_indices

		self.submodels = []
		for i in range(n_models):
			model_path = os.path.join(self.model_dir, f'model_{i}.{self.model_ext}')
			self.train_load_submodel(model_path, cl_ind[i])
			self.submodels.append(model_path)

		if self.call_para('split_models','mix_model'):
			model_path = os.path.join(self.storage_dir, f'model_mix.{self.model_ext}')
			self.train_load_submodel(model_path, None)
			self.mix_model = model_path
		else:
			self.mix_model = None

	def create_temp_dataset(self, cl_ind):

		from sgdml.utils.io import dataset_md5

		name=os.path.join(self.storage_dir,f"temp_dataset.npz")

		data=dict(self.dataset)
		for i in ['R','E','F']:
			data[i]=data[i][cl_ind]
		data['name']=name
		data['md5']=dataset_md5(data)

		np.savez_compressed(name,**data)

		return name

	def train_load_submodel(self, model_path, cl_ind):

		# indices is actually just the 'init' arg when first loading
		# print_subtitle('Training submodel')

		if cl_ind is not None:
			N = len(cl_ind)
			print_ongoing_process(f"Preparing temporary dataset ({N} points)") 

			dataset_path = self.create_temp_dataset(cl_ind)
			dataset_tuple = (dataset_path, np.load(dataset_path))

			print_ongoing_process(f"Preparing temporary dataset ({N} points)")

		else:
			N = len(self.vars[0])
			print_ongoing_process(f"Preparing temporary dataset ({N} points)") 

			dataset_tuple = (self.args['dataset_file'], self.dataset)

			print_ongoing_process(f"Preparing temporary dataset ({N} points)")

		try:
			para = self.call_para('split_models','data_train_func_args')
			para = generate_custom_args(self, para)[0]
			N = para['n_train']

		except:
			N = '?'

		print_ongoing_process(f"Training model ({N} points)")

		self.call_para(
			'split_models','data_train_func',
			args = [self, dataset_tuple, model_path]
			)
		print_ongoing_process(f"Trained model ({N} points)", True)

		original_indices = self.call_para('split_models', 'original_indices')
		if original_indices:
			pass # toad

	def preload_model_predictions(self):

		predicts=[]

		for i in range(len(self.submodels)):
			args=[self, self.submodels[i],
				self.vars[self.call_para('split_models','preload_input_var')],
				self.call_para('split_models','preload_batch_size')]

			pred_i=self.call_para('split_models','preload_predict_func',args=args)
			predicts.append(pred_i)


		predicts=np.array(predicts)
		np.save( os.path.join(self.storage_dir,'pre_predict.npy'),predicts)


		if self.mix_model is not None:
			args=[self, self.mix_model, 'mix',
				self.vars[self.call_para('split_models','preload_input_var')],
				self.call_para('split_models','preload_batch_size')]

			pred=self.call_para('split_models','preload_predict_func',args=args)		
			np.save( os.path.join(self.storage_dir,'mix_pre_predict.npy'),pred)
		else:
			print_warning("No 'mix.npz' model found, cannot preload its prediction")



		## PRINT SOME INFO ABOUT MODEL
		# if self.get_para('train_models','model_info_func') is not None:
		# 	self.call_para('train_models','model_info_func', 
		# 		args = [self])		
		
	def save_command(self):
		super().save_command()

		if self.call_para('split_models', 'classify'):
			ClassificationHandler.save_classifier(self)