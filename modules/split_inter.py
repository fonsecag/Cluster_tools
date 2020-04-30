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

		# put models back out of temp
		self.clear_cluster_dict_temp()

		if self.call_para('split_models', 'classify'):
			self.print_stage('Classification')
			ClassificationHandler.prepare_classifier(self)

		if self.call_para('split_models', 'preload_predict'):
			self.print_stage('Pre-predicting models')
			self.preload_model_predictions()

	def clear_cluster_dict_temp(self):
		self.submodels = []
		for i in range(len(self.cluster_dict)):
			branch = self.cluster_dict[i]
			temp_model = branch['model']
			model = os.path.join(self.model_dir, f'model_{i}.{self.model_ext}')
			shutil.move(temp_model, model)
			self.submodels.append(model)

	def print_branch_summary(self):
		print_subtitle('-------------- Current branch summary --------------')
		print(f"{'N':<3}{'size':<8}{'err':<8}{'rej':<5}{'temp_file'}")

		for i in range(len(self.cluster_dict)):
			b = self.cluster_dict[i]
			print(f"{i:<3}{len(b['ind']):<8}{b['err']:<8.2f}{b['rejected']:<5}"
				f"{b['model']}")
		print('----------------------------------------------------')

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
		return branches

	def attempt_branching(self):

		## FIND ALLOWED BRANCH
		index = self.find_splittable_branch()
		if index is None:
			return False # this exits the while loop

		## SPLIT BRANCH IF FOUND
		branches, valid = self.split_branch(index)

		if valid:
			print_subtitle('Training branch')
			branches = self.train_branch(branches)

			print_subtitle('Checking errors and acceptance criterion')
			branches = self.compute_local_errors(branches)
			success = self.reject_accept_branch(index, branches)
		else:
			success = self.reject_accept_branch(index, branches, False)

		self.print_branch_summary()

		return True

	def find_splittable_branch(self):
		for i in range(len(self.cluster_dict)):
			if not self.cluster_dict[i]['rejected']:
				return i
		return None

	def split_branch(self, index):
		valid = True
		from funcs.cluster import cluster_do
		scheme=self.call_para('fine_clustering','clustering_scheme')

		cl_ind = self.cluster_dict[index]['ind']
		cl_ind_new = cluster_do(self, scheme, cl_ind)
		self.print_cluster_summary(cl_ind_new)

		branch = [{'ind':x, 'rejected':False, 'err':None, 'model':None}
			for x in cl_ind_new]

		# check valid min size
		min_size = self.call_para('split_inter', 'accept_min_size')
		if min_size is not None:
			for x in branch:
				if len(x['ind'])<min_size:
					valid = False
					print('Min size reached, rejected!')
					break

		return branch, valid
		
	def train_branch(self, branches):

		for i in range(len(branches)):
			branch = branches[i]
			if branch['model'] is not None:
				continue

			model = f"model_{self.unique_model_id()}.{self.model_ext}"
			model = os.path.join(self.temp_dir, model)

			self.train_load_submodel(model, branch['ind'])
			branch['model'] = model

		return branches

	def reject_accept_branch(self, index, branches, valid = True):

		if valid:
			new_score = self.call_para('split_inter', 'score_function',
				args = [self, branches])
			old_score = self.call_para('split_inter', 'score_function',
				args = [self, [self.cluster_dict[index]]])

		if (not valid) or (old_score < new_score): # lower score is better
			self.cluster_dict[index]['rejected'] = True
			print_ongoing_process('Rejected branch splitting', True)
			return False 
		else:
			print_ongoing_process('Accepted branch splitting', True)


		print_ongoing_process('Merging tree')
		# accept the branch, integrate it into cluster_dict
		cd = self.cluster_dict

		# first replace the current index 
		cd[index] = branches[0]
		for i in range(1, len(branches)):
			cd.insert(index + i, branches[i])

		# update self.cluster_indices as well just in case
		self.cluster_indices = [ x['ind'] for x in cd]
		print_ongoing_process('Merging tree', True)

	def print_local_errors(self):
		err = self.sample_err 
		ce = self.fine_cluster_err

		summary_table = {}
		summary_table['Min cl. err.'] = f"{np.min(ce):.3f}"
		summary_table['Max cl. err.'] = f"{np.max(ce):.3f}"
		summary_table['Avg cl. err.'] = f"{np.average(ce):.3f}"


		print_table("Fine cluster error summary:",None,None,summary_table, width = 15)

	def save_command(self):
		super().save_command()

		if self.call_para('split_models', 'classify'):
			ClassificationHandler.save_classifier(self)