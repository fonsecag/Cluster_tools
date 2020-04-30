from util import*
import numpy as np
from sgdml.predict import GDMLPredict


def load_sgdml_model(self, path):
	a = np.load(path)
	training_indices = a['idxs_train']
	m = GDMLPredict(a)
	return m, training_indices

def sgdml_all_default(train_indices, args):
	from sgdml.cli import create, train, validate, select, test
	from sgdml.utils import ui, io


	ui.print_step_title('STEP 1', 'Cross-validation task creation')
	task_dir = create(**args)
	dataset = args['dataset'][1]


	if (train_indices is not None) and not (type(train_indices) == int):
		# CHANGE TRAINING INDICES
		# AND RELATED ARRAYS
		R_train = dataset['R'][train_indices]
		F_train = dataset['F'][train_indices]
		E_train = dataset['E'][train_indices]

		for file in os.listdir(task_dir):
			if file.endswith('.npz'):
				name = os.path.join(task_dir, file)
				a = dict(np.load(name, allow_pickle = True))
				a['R_train'] = R_train
				a['F_train'] = F_train
				if 'E_train' in a:
					a['E_train'] = R_train
				a['idxs_train'] = train_indices
				np.savez_compressed(name, **a)


	ui.print_step_title('STEP 2', 'Training')
	task_dir_arg = io.is_dir_with_file_type(task_dir, 'task')
	args['task_dir'] = task_dir_arg
	model_dir_or_file_path = train(**args)



	ui.print_step_title('STEP 3', 'Validation')
	model_dir_arg = io.is_dir_with_file_type(
		model_dir_or_file_path, 'model', or_file=True
	)

	valid_dataset = args['valid_dataset']
	validate(
		model_dir_arg,
		valid_dataset,
		overwrite=False,
		max_processes=args['max_processes'],
		use_torch=args['use_torch'],
	)


	ui.print_step_title('STEP 4', 'Hyper-parameter selection')
	model_file_name = select(
		model_dir_arg, 
		args['overwrite'],
		args['max_processes'],
		args['model_file']
	)


	ui.print_step_title('STEP 5', 'Testing')
	model_dir_arg = io.is_dir_with_file_type(model_file_name, 'model', or_file=True)
	test_dataset = args['test_dataset']

	test(
		model_dir_arg,
		test_dataset,
		args['n_test'],
		overwrite=False,
		max_processes=args['max_processes'],
		use_torch=args['use_torch'],
	)

	print(
		'\n'
		+ ui.color_str('  DONE  ', fore_color=ui.BLACK, back_color=ui.GREEN, bold=True)
		+ ' Training assistant finished sucessfully.'
	)
	print('         This is your model file: \'{}\''.format(model_file_name))

def sgdml_predict_F(self, R):
	model = self.curr_model 
	_, F = model.predict(R)
	return F 

def sgdml_predict_E(self, R):
	model = self.curr_model 
	E, _ = model.predict(R)
	return E 

def sgdml_model_info(self):
	model = self.curr_model
	dic = model.__dict__
	print(f"{'n_atoms':<10}{dic['n_atoms']}")
	print(f"{'n_train':<10}{dic['n_train']}")

def sgdml_train_default(self, train_indices, model_path, 
		old_model_path, sgdml_args):


	args=sgdml_args.copy()
	dataset_tuple = (self.args['dataset_file'], self.dataset)
	if type(train_indices)==int:
		n_train = train_indices
	else:
		n_train = len(train_indices)

	if (old_model_path is not None):
		model0 = (old_model_path, np.load(old_model_path, allow_pickle = True))
	else:
		model0 = None

	task_dir=os.path.join(self.storage_dir,f"sgdml_task_{n_train}")
	args['n_train']=n_train
	args['task_dir']=os.path.join(self.storage_dir,f"sgdml_task_{n_train}")
	args['valid_dataset']=dataset_tuple
	args['test_dataset']=dataset_tuple
	args['dataset']=dataset_tuple
	args['model_file']=model_path
	args['command']='all'
	args['max_processes']=self.n_cores
	args['model0'] = model0

	if self.call_para('train_models','suppress_sgdml_prints'):
		with sgdml_print_suppressor():
			sgdml_all_default(train_indices, args)

	else:
		sgdml_all_default(train_indices, args)


	# if self.call_para('train_models','suppress_sgdml_prints'):
	# 	with sgdml_print_suppressor():
	# 		cli.all(**args)
	# else:
	# 	cli.all(**args)


def sgdml_train_data(self, dataset_tuple, model_path, sgdml_args):


	args=sgdml_args.copy()

	task_dir=os.path.join(self.storage_dir,f"sgdml_task_temp")

	args['task_dir'] = task_dir
	args['valid_dataset'] = dataset_tuple
	args['test_dataset'] = dataset_tuple
	args['dataset'] = dataset_tuple
	args['model_file'] = model_path
	args['command'] = 'all'
	args['max_processes'] = self.n_cores
	args['model0'] = None

	if self.call_para('train_models','suppress_sgdml_prints'):
		with sgdml_print_suppressor():
			sgdml_all_default(None, args)

	else:
		sgdml_all_default(None, args)

def sgdml_path_predict_F(self, model_path, input_var, batch_size):
	from sgdml.predict import GDMLPredict

	N=len(input_var)
	n_batches=N//batch_size+1

	if n_batches>999:
		width=20
	else:
		width=None

	npz = np.load(model_path)
	model=GDMLPredict(npz)


	message=f'Predicting {os.path.basename(model_path)} batches'

	predicts=[]

	start_time,eta=time.time(),0
	for i in range(n_batches):
		print_x_out_of_y_eta(message,i,n_batches,eta,width=width)
		R=input_var[i*batch_size:(i+1)*batch_size]
		if len(R)==0:
			break
		_,F=model.predict(R)
		predicts.append(F)

		avg_time=(time.time()-start_time)/(i+1)
		eta=(n_batches-i+1)*avg_time

	print_x_out_of_y_eta(message,n_batches,n_batches,time.time()-start_time,True,width=width)

	predicts=np.concatenate(predicts)
	return predicts


# def sgdml_path_predict_F_ind(self, model_path, id, input_var, ind, batch_size):
# 	from sgdml.predict import GDMLPredict

# 	N=len(ind)
# 	n_batches=N//batch_size+1

# 	if n_batches>999:
# 		width=20
# 	else:
# 		width=None

# 	input_var = np.array(input_var)[ind]

# 	npz = np.load(model_path)
# 	model=GDMLPredict(npz)

# 	message=f'Predicting model_{id} batches'

# 	predicts=[]

# 	start_time,eta=time.time(),0
# 	for i in range(n_batches):
# 		print_x_out_of_y_eta(message,i,n_batches,eta,width=width)
# 		R=input_var[i*batch_size:(i+1)*batch_size]
# 		if len(R)==0:
# 			break
# 		_,F=model.predict(R)
# 		predicts.append(F)

# 		avg_time=(time.time()-start_time)/(i+1)
# 		eta=(n_batches-i+1)*avg_time

# 	print_x_out_of_y_eta(message,n_batches,n_batches,time.time()-start_time,True,width=width)

# 	predicts=np.concatenate(predicts)
# 	return predicts


def get_sgdml_training_set(self, model):
	print(model.__dict__.keys())
	sys.exit()


### mbGDMLPredict ###

class solvent():
	
	def __init__(self, atom_list):
	  """Identifies and describes the solvent for MB-GDML.
	
	  Represents the solvent that makes up the cluster MB-GDML is being trained
	  from.
	  
	  Attributes:
		  system (str): Designates what kind of system this is. Currently only
			  'solvent' systems.
		  solvent_name (str): Name of the solvent.
		  solvent_label (str): A label for the solvent for filenaming purposes.
		  solvent_molec_size (int): Total number of atoms in one solvent molecule.
		  cluster_size (int): Total number of solvent molecules in cluster.
	  """

	  # Gets available solvents from json file.
	  # solvent_json = '../solvents.json'
	  # with open(solvent_json, 'r') as solvent_file:
		 #  solvent_data=solvent_file.read()
	  all_solvents = {
		    "water": {
		        "label": "H2O",
		        "formula": "H2O"
		    },
		    "acetonitrile": {
		        "label": "acn",
		        "formula": "C2H3N"
		    },
		    "methanol": {
		        "label": "MeOH",
		        "formula": "CH3OH"
		    }
		}
	  
	  self.identify_solvent(atom_list, all_solvents)


	def atom_numbers(self, chem_formula):
		"""Provides a dictionary of atoms and their quantity from chemical
		formula.
		
		Args:
			chem_formula (str): the chemical formula of a single solvent
				molecule. For example, 'CH3OH' for methanol, 'H2O' for water,
				and 'C2H3N' for acetonitrile.
		
		Returns:
			dict: contains the atoms as their elemental symbol for keys and
				their quantity as values.
		
		Example:
			atom_numbers('CH3OH')
		"""
		string_list = list(chem_formula)
		atom_dict = {}
		str_index = 0
		while str_index < len(string_list):
			next_index = str_index + 1
			if string_list[str_index].isalpha():
				# Checks to see if there is more than one of this atom type.
				try:
					if string_list[next_index].isalpha():
						number = 1
						index_change  = 1
					elif string_list[next_index].isdigit():
						number = int(string_list[next_index])
						index_change  = 2
				except IndexError:
					number = 1
					index_change  = 1

			# Adds element numbers to atom_dict
			if string_list[str_index] in atom_dict:
				atom_dict[string_list[str_index]] += number
			else:
				atom_dict[string_list[str_index]] = number
			
			str_index += index_change

		return atom_dict


	def identify_solvent(self, atom_list, all_solvents):
		"""Identifies the solvent from a repeated list of elements.
		
		Args:
			atom_list (lst): List of elements as strings. Elements should be
				repeated. For example, ['H', 'H', 'O', 'O', 'H', 'H']. Note that
				the order does not matter; only the quantity.
			all_solvents (dict): Contains all solvents described in
				solvents.json. Keys are the name of the solvent, and the values
				are dicts 'label' and 'formula' that provide a solvent label
				and chemical formula, respectively.
		
		"""

		# Converts atoms identified by their atomic number into element symbols
		# for human redability.
		from periodictable import elements

		if str(atom_list[0]).isdigit():
			atom_list_elements = []
			for atom in atom_list:
				atom_list_elements.append(str(elements[atom]))
			atom_list = atom_list_elements

		# Determines quantity of each element in atom list.
		# Example: {'H': 4, 'O': 2}
		atom_num = {}
		for atom in atom_list:
			if atom in atom_num.keys():
				atom_num[atom] += 1
			else:
				atom_num[atom] = 1
		
		# Identifies solvent by comparing multiples of element numbers for each
		# solvent in the json file. Note, this does not differentiate
		# between isomers or compounds with the same chemical formula.
		# Loops through available solvents and solvent_atoms.
		for solvent in all_solvents:
			
			solvent_atoms = self.atom_numbers(all_solvents[solvent]['formula'])
			# Number of elements in atom_list should equal that of the solvent.
			if len(atom_num) == len(solvent_atoms):

				# Tests all criteria to identify solvent. If this fails, it moves
				# onto the next solvent. If all of them fail, it raises an error.
				try:
					# Checks that the number of atoms is a multiple of the solvent.
					# Also checks that the multiples of all the atoms are the same.
					solvent_numbers = []

					# Checks that atoms are multiples of a solvent.
					for atom in solvent_atoms:
						multiple = atom_num[atom] / solvent_atoms[atom]
						if multiple.is_integer():
							solvent_numbers.append(multiple)
						else:
							raise ValueError

					# Checks that all multiples are the same.
					test_multiple = solvent_numbers[0]
					for multiple in solvent_numbers:
						if multiple != test_multiple:
							raise ValueError
					
					
					self.solvent_name = str(solvent)
					self.solvent_label = str(all_solvents[solvent]['label'])
					self.solvent_molec_size = int(sum(solvent_atoms.values()))
					self.cluster_size = int(atom_num[atom] \
											/ solvent_atoms[atom])
					self.system = 'solvent'  # For future
					# developments that could involve solute-solvent systems.
					
				except:
					pass

		if not hasattr(self, 'solvent_name'):
			print('The solvent could not be identified.')

def system_info(atoms):
	"""Determines information about the system key to mbGDML.
	
	Args:
		atoms (list): Atomic numbers of all atoms in the system.
	
	Returns:
		dict: System information useful for mbGDML. Information includes
			'system' which categorizes the system and provides specific
			information.
	
	Notes:
		For a 'solvent' system the additional information returned is the
		'solvent_name', 'solvent_label', 'solvent_molec_size', and
		'cluster_size'.
	"""
	
	system_info = solvent(atoms)
	system_info_dict = {'system': system_info.system}
	if system_info_dict['system'] is 'solvent':
		system_info_dict['solvent_name'] = system_info.solvent_name
		system_info_dict['solvent_label'] = system_info.solvent_label
		system_info_dict['solvent_molec_size'] = system_info.solvent_molec_size
		system_info_dict['cluster_size'] = system_info.cluster_size
	
	return system_info_dict

class mbGDMLPredict():

	def __init__(self, mb_gdml):
		"""Sets GDML models to be used for many-body prediction.
		
		Args:
			mb_gdml (list): Contains sgdml.GDMLPredict objects for all models
				to be used for prediction.
		"""
		self.gdmls = mb_gdml


	def predict(self, z, R):
		
		if len(R.shape)>2:
			return self.predict_bulk(z, R)

		# Gets system information from dataset.
		# This assumes the system is only solvent.
		dataset_info = system_info(z.tolist())
		system_size = dataset_info['cluster_size']
		molecule_size = dataset_info['solvent_molec_size']

		# Sets up arrays for storing predicted energies and forces.
		F = np.zeros(R.shape)
		E = 0.0

		# Adds contributions from all models.
		for gdml in self.gdmls:
			
			model_atoms = int(gdml.n_atoms)
			nbody_order = int(model_atoms/molecule_size)
			
			# Getting list of n-body combinations.
			nbody_combinations = list(
				itertools.combinations(list(range(0, system_size)), nbody_order)
			)

			# Adding all contributions.
			for comb in nbody_combinations:
				# Gets indices of all atoms in the
				# n-body combination of molecules.
				atoms = []
				for molecule in comb:
					atoms += list(range(
						molecule * molecule_size, (molecule + 1) * molecule_size
					))
				
				# Adds n-body contributions prediced from nbody_model.
				e, f = gdml.predict(R[atoms].flatten())
				F[atoms] += f.reshape(len(atoms), 3)
				E += e
		
		return E, F

	def predict_bulk(self, z, R):

		dataset_info = system_info(z.tolist())
		system_size = dataset_info['cluster_size']
		molecule_size = dataset_info['solvent_molec_size']

		# Sets up arrays for storing predicted energies and forces.
		F = np.zeros(R.shape)
		n_samples = len(R)
		E = np.zeros(n_samples)

		# Adds contributions from all models.
		for gdml in self.gdmls:
			
			model_atoms = int(gdml.n_atoms)
			nbody_order = int(model_atoms/molecule_size)
			
			# Getting list of n-body combinations.
			nbody_combinations = list(
				itertools.combinations(list(range(0, system_size)), nbody_order)
			)

			# Adding all contributions.
			for comb in nbody_combinations:
				# Gets indices of all atoms in the
				# n-body combination of molecules.
				atoms = []
				for molecule in comb:
					atoms += list(range(
						molecule * molecule_size, (molecule + 1) * molecule_size
					))
				
				# Adds n-body contributions prediced from nbody_model.
				e, f = gdml.predict(R[:,atoms].reshape(n_samples, -1))
				F[:,atoms] += f.reshape(n_samples, len(atoms), 3)
				E += e
		
		return E, F		

def load_mbGDML_model(self, path):
	n_mers = self.call_para('train_models', 'n_mers')

	if not os.path.isdir(path):
		print_error('Model path (-i) needs to be a directory for mbGDML models'\
			f'. Path given: {path}')

	# find the files automatically based on the naming scheme
	# *1mer*npz, *2body*npz, *3body*npz ... *nbody*npz
	mb_gdmls = [None]*n_mers
	for file in glob.glob(f'{path}/*.npz'):

		a = re.match(".*(\d)body.*npz|.*(1)mer.*npz", file)
		if a is not None:

			n1, n2 = a.groups()
			if n1 is None:
				n = int(n2) -1
			else:
				n = int(n1) -1

			if n<n_mers:
				mb_gdmls[n] = GDMLPredict(np.load(file))

	model = mbGDMLPredict(mb_gdmls)

	# # test
	# r = self.vars[1][:5]
	# _,a = model.predict(self.dataset['z'], r[0])
	# _,b = model.predict(self.dataset['z'], r)
	# print( a - b[0])

	return model, [] # preliminarily empty training indices array

def mbgdml_predict_F(self, R):
	model = self.curr_model
	z = self.dataset['z']
	n_samples = len(R)
	_, F = model.predict(z, R)
	return F.reshape(n_samples, -1)