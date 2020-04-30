from util import*
import data_handling
from funcs import cluster, classification, misc, models

def parse_arguments(args):

    full_call=" ".join(args)

    p=argparse.ArgumentParser(
        prog=None,
        description="Cluster dataset and create submodels for each.",
        )

    # ADD SUBPARSERS
    subp = p.add_subparsers(title='commands', dest='command', help=None)
    subp.required = True

    p_cluster = subp.add_parser(
        'cluster', 
        help='cluster dataset',
    )

    p_class = subp.add_parser(
        'class',
        help='cluster data and classify',
    )

    p_train = subp.add_parser(
        'train', 
        help='train improved model',
    )

    p_cluster_error = subp.add_parser(
        'cluster_error',
        help='Cluster dataset, calculate and plot errors of given model',
    )

    p_plot_cluster_error = subp.add_parser(
        'plot_error',
        help='Plot the errors for given info file',
    )

    p_split_train = subp.add_parser(
        'split_train',
        help='Train a split model',
    )

    p_split_inter = subp.add_parser(
        'split_inter',
        help='Interactive split training. EXPERIMENTAL.',
    )

    p_cluster_xyz = subp.add_parser(
        'xyz',
        help = 'Write xyz files for every cluster'
    )

    subp_all=[p_cluster, p_class, p_train, p_cluster_error, p_cluster_xyz, 
        p_split_train, p_split_inter]

    # all except p_plot_cluster_error
    
    # ADD ARGUMENTS FOR ALL
    for x in subp_all:
        x.add_argument(
            '-d',
            '--dataset',
            metavar='<dataset_file>',
            dest='dataset_file',
            help='path to dataset file',
            required=True,
            )

        x.add_argument(
            '-p',
            '--para',
            metavar='<para_file>',
            dest='para_file',
            help="name of para file",
            required=False,
            default='default',
            )

        x.add_argument(
            '-c',
            '--cluster',
            metavar='<cluster_file>',
            dest='cluster_file',
            help='path to cluster file',
            default=None,
           )

    # ADD ARGUMENTS FOR INFO-DEPENDENTS
    info_boys = [p_plot_cluster_error]
    for x in info_boys:
        x.add_argument(
            '-i',
            '--info',
            metavar = '<info_file>',
            dest='info_file',
            help='path to info file',
            required=True,
            )

        x.add_argument(
            '-p',
            '--para',
            metavar='<para_file>',
            dest='para_file',
            help="name of para file",
            required=False,
            default='default',
            )


    # ADD SPECIFIC ARGUMENTS FOR SUBS
    for x in [p_train]:
        x.add_argument(
            '-n',
            '--n_steps',
            metavar='<n_steps>',
            dest='n_steps',
            help='Number of steps',
            required=True,
            type = int,
            )

        x.add_argument(
            '-s',
            '--size',
            metavar='<step_size>',
            dest='step_size',
            help='Step size (in number of points)',
            required=True,
            type = int,
            )

        x.add_argument(
            '-i',
            '--init',
            metavar='<init>',
            dest='init',
            help='Initial model (path) or initial number of points (int)',
            required=True,
            )

    for x in [p_cluster_error]:
        x.add_argument(
            '-i',
            '--init',
            metavar='<init>',
            dest='init',
            help='Initial model to calculate errors of',
            required=True,
        )

    # PARSE
    args=p.parse_args()
    args=vars(args)

    # HANDLE ARGS


    # find para file
    sys.path.append("paras")
    para_file=args['para_file']

    para_file=para_file.replace(".py","") #in case the user includes '.py' in the name
    para_file=os.path.basename(para_file) 

    file=os.path.join("paras",para_file)

    if os.path.exists(file+".py"): 
        args['para_file']=para_file
    else:
        print_error(f"No valid para file found under name: {args['para_file']}")

    # add full call
    args['full_call']=full_call

    return args


class MainHandler():

    def __init__(self,args, needs_dataset = True):

        self.args=args
        #load para filer_to_dist
        para_file=args['para_file']
        para_mod=__import__(para_file)
        para=para_mod.parameters
        funcs=func_dict_from_module(para_mod)

        # merge with defaults 
        para_def_mod=__import__("default")
        para_def=para_def_mod.parameters

        # extracts any function from the para.py file
        funcs_def=func_dict_from_module(para_def_mod)

        merge_para_dicts(para_def,para) #WAI

        self.para=para_def
        z={**funcs_def,**funcs} #WAI
        self.funcs=z

        n_cores = int(self.call_para('n_cores') or 1)
        if n_cores==0:
            n_cores = 1
        elif n_cores<0:
            n_cores = os.cpu_count()+n_cores
        self.n_cores = n_cores

        # merge exceptions
        if para.get('load_dataset',{}).get('var_funcs',None) is not None:
            self.para['load_dataset']['var_funcs']=para['load_dataset']['var_funcs']

        if not needs_dataset:
            self.n_main_stages -= 2 
            self.needs_dataset = False


    needs_dataset = True
    vars=[]
    info = {}

    SEARCH_MODULES=[cluster,data_handling,classification,misc,models]
    def find_function(self,name):
        return find_function(name,self.funcs,self.SEARCH_MODULES)

    def generate_para_args(self,args):
        return generate_custom_args(self,args)

    def generate_para_kwargs(self, kwargs):
        return generate_custom_kwargs(self, kwargs)

    def call_para(self,*path,args=[], kwargs={}):
        if len(path)==0:
            return None

        para=self.para
        subdict=para
        step=para


        for x in path:
            subdict=step
            step=step.get(x,None)


        # handle functions
        if type(step)==str and step.startswith("func:"):
            f_name=step.replace("func:","")
            f=self.find_function(f_name)

            arg_name=str(path[-1])+'_args'
            args_add=self.generate_para_args(subdict.get(arg_name,[]))

            kwarg_name=str(path[-1])+'_kwargs'
            kwargs_add=self.generate_para_kwargs(subdict.get(kwarg_name,{}))

            kwargs.update(kwargs_add)

            args_full=args+args_add
            return f(*args_full, **kwargs)

        elif type(step)==str and step.startswith("para:"):
            new_args=step.replace("para:","").split(",")
            return self.call_para(*newargs,args=args)

        elif step is None:
            return None
        else:
            # Not needed any more, call_para is more versatile and so is the default now
            # print_warning(f"Tried to call para: {path}={step}, but not callable. Value returned instead.")
            return step

    def return_partial_func(self, *path, kwargs = {}):
        if len(path)==0:
            return None

        para=self.para
        subdict=para
        step=para

        for x in path:
            subdict=step
            step=step.get(x,None)


        # handle functions
        if type(step)==str and step.startswith("func:"):
            f_name=step.replace("func:","")
            f=self.find_function(f_name)
            kwarg_name=str(path[-1])+'_kwargs'
            kwargs_add=self.generate_para_kwargs(subdict.get(kwarg_name,{}))
            kwargs.update(kwargs_add)
        else:
            print_error(f'Para {path} not a function')

        func = self.get_para(*path)
        return partial(func, **kwargs)

    def get_para(self,*path,args=[]):

        if len(path)==0:
            return None

        para=self.para
        step=para

        for x in path:
            step=step.get(x,None)

        # handle functions
        if type(step)==str and step.startswith("func:"):
            f_name=step.replace("func:","")
            f=self.find_function(f_name)
            return f
        elif type(step)==str and step.startswith("para:"):
            new_args=step.replace("para:","").split(",")
            return self.get_para(*newargs,args=args)
        else:
            return step

    def print_stage(self, s):
        print_stage(s, self.current_stage, self.n_stages)
        self.current_stage += 1

    current_stage = 1 
    n_main_stages = 4
    def run(self):

        if self.needs_dataset:
            self.print_stage('Load dataset')
            self.load_dataset()

            self.print_stage('Prepare vars')
            self.prepare_vars()

        self.print_stage('Prepare storage')
        self.prepare_storage()

        self.run_command()

        self.print_stage('Save in storage')
        self.save_main()
        self.save_command()
        self.delete_temp()

    def delete_temp(self):
        shutil.rmtree(self.temp_dir)

    def load_dataset(self):
        path=self.args['dataset_file']

        if path is None:
            print_error(f"No dataset given. Please use the -d arg followed by the path to the dataset.")
        elif not os.path.exists(path):
            print_error(f"Dataset path {path} is not valid.")

        ext=os.path.splitext(path)[-1]
        #xyz file
        if ext==".xyz":
            print_ongoing_process(f"Loading xyz file {path}")
            try:
                file=open(path)
                dat=read_concat_ext_xyz(file)
                data={ 'R':np.array(dat[0]),'z':dat[1],'E':np.reshape( dat[2] , (len(dat[2]),1) ),'F':np.array(dat[3]) }
            except Exception as e:
                print(e)
                print_error("Couldn't load .xyz file.")

            print_ongoing_process(f"Loaded xyz file {path}",True)

        #npz file        
        elif ext==".npz":
            print_ongoing_process(f"Loading npz file {path}")
            try:
                data=np.load(path,allow_pickle=True)
            except Exception as e:
                print(e)
                print_error("Couldn't load .npz file.")

            print_ongoing_process(f"Loaded npz file {path}",True)

        else:
            print_error(f"Unsupported data type {ext} for given dataset {path} (xyz and npz supported).")

        
        self.dataset=data
        if self.get_para('load_dataset','post_processing') is not None:
            print_ongoing_process('Post-processing dataset')
            self.call_para('load_dataset','post_processing',args=[self])
            print_ongoing_process('Post-processing dataset',True)

    def prepare_vars(self):

        dataset=self.dataset

        #get the needed vars ready
        #parses through data set and uses the given functions to generate the needed variables
        #f.e. interatomic distances and energies
        var_funcs=self.call_para('load_dataset','var_funcs')
        keys=list(var_funcs.keys())
        for i in range(len(keys)):
            print_x_out_of_y("Extracting vars",i,len(keys))
            x=keys[i]
            self.vars.append(self.call_para("load_dataset","var_funcs",x,args=[self.dataset]))
        print_x_out_of_y("Extracting vars",len(keys),len(keys),True)

        # SUMMARY
        summary_table={}
        for i in range(len(self.vars)):
            try:
                summary_table[i]=self.vars[i].shape
            except:
                summary_table[i]="No shape"
        print_table("Vars summary:","index","shape",summary_table)

    def do_nothing(*args):
        print_debug("Doing nothing. Please be patient.")

    def prepare_storage(self):

        print_ongoing_process("Preparing save directory")
        storage_dir=self.call_para('storage','storage_dir')
        dir_name=f"{self.args['command']}_{self.call_para('storage','dir_name')}"

        path=find_valid_path(os.path.join(storage_dir,dir_name))
        self.storage_dir=path

        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print_warning(f"Save path {path} already exists. How? Overwriting of files possible.")
        print_ongoing_process(f"Prepared save directory {path}",True)

        # copy user para file
        if self.call_para('storage','save_para_user'):
            print_ongoing_process("Saving user para file")
            file_name=self.args.get("para_file")+".py"
            file=os.path.join("paras",file_name)
            if os.path.exists(file):
                shutil.copy(file,os.path.join(path,file_name))
                print_ongoing_process(f"Saved user para file {os.path.join(path,file_name)}",True)
            else:
                print_warning(f"Tried copying user parameter file {file}. Not found")

        # copy default para file
        if self.call_para('storage','save_para_default'):
            print_ongoing_process('Saving default para file')
            file_name="default.py"
            file=os.path.join("paras",file_name)
            if os.path.exists(file):
                shutil.copy(file,os.path.join(path,file_name))
                print_ongoing_process(f'Saved default para file {os.path.join(path,file_name)}',True)
            else:
                print_warning(f"Tried copying default parameter file {file}. Not found")

        if self.call_para('storage','save_original_call'):
            print_ongoing_process('Saving original call')
            with open(os.path.join(path,"Call.txt"),'w+') as file:
                print(f"Original call: {self.args.get('full_call','N/A')}",file=file)
                print_ongoing_process(f'Saved original call at {os.path.join(path,"Call.txt")}',True)

        # create temp folder
        self.temp_dir = os.path.join(self.storage_dir, 'temp')
        os.mkdir(self.temp_dir)

    def save_main(self):
        self.info['para'] = self.para
        self.info['args'] = self.args

        print_ongoing_process('Saving info file')
        info_path = os.path.join(self.storage_dir,'info.p')
        with open(info_path,'wb') as file:
            pickle.dump(self.info,file)
        print_ongoing_process('Saved info file', True)

    def load_info_file(self, path):

        print_ongoing_process('Loading info file')
        with open(path,'rb') as file:
            info = pickle.loads(file.read())

        self.info = info
        if 'cluster_indices' in info:
            self.cluster_indices = info['cluster_indices']

        if 'errors' in info:
            self.errors = info['errors']

        info['args'] = self.args
        print_ongoing_process('Loaded info file', True)
        summary_table = {}
        for k,v in info.items():
            summary_table[k] = f'{type(v)}'
        print_table("Items found:","Key","Value",summary_table, width = 22)

if __name__=='__main__':

    args=parse_arguments(sys.argv[1:])
    command = args['command']

    if command == 'cluster':
        from modules.cluster import ClusterHandler 
        hand = ClusterHandler(args)
        
    elif command == 'class':
        from modules.classification import ClassificationHandler
        hand = ClassificationHandler(args)

    elif command == 'train':
        from modules.train import TrainHandler 
        hand = TrainHandler(args)

    elif command == 'cluster_error':
        from modules.clustererror import ClusterErrorHandler 
        hand = ClusterErrorHandler(args)

    elif command == 'plot_error':
        from modules.plotclustererror import PlotClusterErrorHandler 
        hand = PlotClusterErrorHandler(args, needs_dataset = False)

    elif command == 'xyz':
        from modules.cluster_xyz import ClusterXYZHandler
        hand = ClusterXYZHandler(args)

    elif command == 'split_train':
        from modules.split_model import SplitModelHandler
        hand = SplitModelHandler(args)
        
    elif command == 'split_inter':
        from modules.split_inter import SplitInterHandler
        hand = SplitInterHandler(args)

    # actually run the friggin' thing
    hand.run()

    print_successful_exit("run.py exited successfully")