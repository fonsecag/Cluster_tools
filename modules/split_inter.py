from run import MainHandler
from .split_model import SplitModelHandler
from .cluster import ClusterHandler
from .classification import ClassificationHandler
from util import*


class SplitInterHandler(SplitModelHandler):

	def __init__(self,args, **kwargs):
		super().__init__(args, **kwargs)
		self.n_stages += 1

	def run_command(self):

		self.model_dir = os.path.join(self.storage_dir, 'models')
		self.model_ext = self.call_para('train_models','model_ext')
		os.mkdir(self.model_dir)

		ClusterHandler.run_command(self)
		self.cluster_dict = [ 
			{'ind':np.copy(x), 'rejected':False, 'err':None, 'model':None} 
			for x in self.cluster_indices]


		## INITIAL MODEL
		self.print_stage('Creating initial submodels')
		self.create_split_models()

		print_ongoing_process('Copying models to temporary folder')
		# move the created models to temp
		for i in range(len(self.cluster_indices)):
			ori = os.path.join(self.model_dir, f'model_{i}.{self.model_ext}')
			if not os.path.exists(ori):
				print_error(f'Model file {ori} not found.'
					+' Did the model naming scheme change?')

			model_id, ext = self.unique_model_id(), self.model_ext
			new = os.path.join(self.temp_dir, f'model_{model_id}.{ext}')

			shutil.move(ori, new)
			self.cluster_dict[i]['model'] = new
		print_ongoing_process('Copying models to temporary folder', True)

		self.compute_local_errors()
		self.print_branch_summary()

		## BRANCHING
		self.print_stage('Branching')
		iters = 0
		while self.attempt_branching():
			iters += 1
			if iters > 100:
				print_warning('More than 100 branching iterations, broke out')
				break

		if self.call_para('split_models', 'classify'):
			self.print_stage('Classification')
			ClassificationHandler.prepare_classifier(self)

		if self.call_para('split_models', 'preload_predict'):
			self.print_stage('Pre-predicting models')
			self.preload_model_predictions()


	def print_branch_summary(self):
		print_subtitle('Current branch summary')
		print(f"{'N':<3}{'size':<8}{'err':<8}{'rej':<5}{'temp_file'}")

		for i in range(len(self.cluster_dict)):
			b = self.cluster_dict[i]
			print(f"{i:<3}{len(b['ind']):<8}{b['err']:<8.2f}{b['rejected']:<5}"
				f"{b['model']}")

	unique_id = 0
	def unique_model_id(self):
		self.unique_id += 1
		return self.unique_id

	def compute_local_errors(self, branches = None):
		print_subtitle('Computing local errors')
		if branches is None:
			branches = self.cluster_dict

		for branch in branches:
			if branch['err'] is not None:
				continue 

			cl_ind = branch['ind']
			inp_index = self.call_para('split_inter', 'local_error_input_index')
			comp_index = self.call_para('split_inter', 'local_error_comp_index')

			args = [self, branch['model'], self.vars[inp_index][cl_ind],
				self.call_para('split_inter', 'local_batch_size')]
			pred = self.call_para('split_inter', 'local_predict_func',
				args = args)

			args = [self, pred, self.vars[comp_index][cl_ind]]
			err = self.call_para('split_inter', 'local_error_func',
				args = args)

			branch['err'] = err
		print_ongoing_process('Computed local errors', True)



	def attempt_branching(self):

		## FIND ALLOWED BRANCH
		index = self.find_splittable_branch()
		if index is None:
			return False # this exits the while loop

		## SPLIT BRANCH IF FOUND
		branches = self.split_branch(index)

		print_subtitle('Training branch')
		branches = self.train_branch(branches)

		print_subtitle('Checking errors and acceptance criterion')
		branch = self.compute_local_errors(branches)
		success = self.reject_accept_branch(index, branch)

		return True

	def find_splittable_branch(self):
		for i in range(len(self.cluster_dict)):
			if not self.cluster_dict[i]['rejected']:
				return i
		return None

	def split_branch(self, index):

		from funcs.cluster import cluster_do
		scheme=self.call_para('fine_clustering','clustering_scheme')

		cl_ind = self.cluster_dict[index]['ind']
		cl_ind_new = cluster_do(self, scheme, cl_ind)
		self.print_cluster_summary(cl_ind_new)

		branch = [{'ind':x, 'rejected':False, 'err':None, 'model':None}
			for x in cl_ind_new]
		return branch
		
	def train_branch(self, branches):

		for i in range(len(branches)):
			branch = branches[i]
			if branch['model'] is not None:
				continue

			model = f"model_{self.unique_model_id()}.{self.model_ext}"
			model = os.path.join(self.temp_dir, model)

			self.train_load_submodel(model, branch['ind'])
			branch['model'] = model

	def reject_accept_branch(self, branches):
		

	def print_local_errors(self):
		err = self.sample_err 
		ce = self.fine_cluster_err

		summary_table = {}
		summary_table['Min cl. err.'] = f"{np.min(ce):.3f}"
		summary_table['Max cl. err.'] = f"{np.max(ce):.3f}"
		summary_table['Avg cl. err.'] = f"{np.average(ce):.3f}"


		print_table("Fine cluster error summary:",None,None,summary_table, width = 15)




	# def create_split_models(self):
	# 	n_models = len(self.cluster_indices)
	# 	cl_ind = self.cluster_indices

	# 	self.submodels = []
	# 	for i in range(n_models):
	# 		model_path = os.path.join(self.model_dir, f'model_{i}.{self.model_ext}')
	# 		self.train_load_submodel(model_path, cl_ind[i])
	# 		self.submodels.append(model_path)

	# 	if self.call_para('split_models','mix_model'):
	# 		model_path = os.path.join(self.storage_dir, f'model_mix.{self.model_ext}')
	# 		self.train_load_submodel(model_path, None)
	# 		self.mix_model = model_path
	# 	else:
	# 		self.mix_model = None

	# def create_temp_dataset(self, cl_ind):

	# 	from sgdml.utils.io import dataset_md5

	# 	name=os.path.join(self.storage_dir,f"temp_dataset.npz")

	# 	data=dict(self.dataset)
	# 	for i in ['R','E','F']:
	# 		data[i]=data[i][cl_ind]
	# 	data['name']=name
	# 	data['md5']=dataset_md5(data)

	# 	np.savez_compressed(name,**data)

	# 	return name

	# def train_load_submodel(self, model_path, cl_ind):

	# 	# indices is actually just the 'init' arg when first loading
	# 	print_subtitle('Training submodel')

	# 	if cl_ind is not None:
	# 		N = len(cl_ind)
	# 		print_ongoing_process(f"Preparing temporary dataset ({N} points)") 

	# 		dataset_path = self.create_temp_dataset(cl_ind)
	# 		dataset_tuple = (dataset_path, np.load(dataset_path))

	# 		print_ongoing_process(f"Preparing temporary dataset ({N} points)")

	# 	else:
	# 		N = len(self.vars[0])
	# 		print_ongoing_process(f"Preparing temporary dataset ({N} points)") 

	# 		dataset_tuple = (self.args['dataset_file'], self.dataset)

	# 		print_ongoing_process(f"Preparing temporary dataset ({N} points)")

	# 	try:
	# 		para = self.call_para('split_models','data_train_func_args')
	# 		para = generate_custom_args(self, para)[0]
	# 		N = para['n_train']

	# 	except:
	# 		N = '?'

	# 	print_ongoing_process(f"Training model ({N} points)")

	# 	self.call_para(
	# 		'split_models','data_train_func',
	# 		args = [self, dataset_tuple, model_path]
	# 		)
	# 	print_ongoing_process(f"Trained model ({N} points)", True)

	# 	original_indices = self.call_para('split_models', 'original_indices')
	# 	if original_indices:
	# 		pass # toad

	# def preload_model_predictions(self):

	# 	predicts=[]

	# 	for i in range(len(self.submodels)):
	# 		args=[self, self.submodels[i], i,
	# 			self.vars[self.call_para('split_models','preload_input_var')],
	# 			self.call_para('split_models','preload_batch_size')]

	# 		pred_i=self.call_para('split_models','preload_predict_func',args=args)
	# 		predicts.append(pred_i)


	# 	predicts=np.array(predicts)
	# 	np.save( os.path.join(self.storage_dir,'pre_predict.npy'),predicts)


	# 	if self.mix_model is not None:
	# 		args=[self, self.mix_model, 'mix',
	# 			self.vars[self.call_para('split_models','preload_input_var')],
	# 			self.call_para('split_models','preload_batch_size')]

	# 		pred=self.call_para('split_models','preload_predict_func',args=args)		
	# 		np.save( os.path.join(self.storage_dir,'mix_pre_predict.npy'),pred)
	# 	else:
	# 		print_warning("No 'mix.npz' model found, cannot preload its prediction")	
		
	def save_command(self):
		super().save_command()

		if self.call_para('split_models', 'classify'):
			ClassificationHandler.save_classifier(self)