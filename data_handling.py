from util import*
from scipy.spatial.distance import pdist

def toDistance(R):
    '''
    This function takes a numpy array containing positions and returns it as distances.
    
    Parameters:
        -R:
            numpy array containing positions for every atom in every sample
            Dimensions: (n_samples,n_atoms,n_dimensions)
            
    Returns:
        -y:
            numpy array containing distances for every atom in every sample
            Dimensions: (n_samples,n_atoms*(n_atoms-1)/2)
    '''
    
    shape=R.shape
    try:
        dim=shape[2]
    except:
        return
    if shape[1]<2:
        return

    y=[]

    for i in range(len(R)): ##goes through samples
        y.append(pdist(R[i]))

    y=np.array(y)
    return y

def r_to_inv_dist(dataset):
    R=dataset['R']
    return 1. / toDistance(R)

def r_to_dist(dataset):
    R=dataset['R']
    return toDistance(R)

def extract_E(dataset):
    E=dataset['E']
    return np.array(E).reshape(-1,1)

def extract_E_neg(dataset):
    E=dataset['E']
    return -np.array(E).reshape(-1,1)

def extract_R_concat(dataset):

    R=dataset['R']
    n_samples,n_atoms,n_dim=R.shape
    R=np.reshape(R,(n_samples,n_atoms*n_dim))
    return np.array(R)

def extract_R(dataset):

    R=dataset['R']
    n_samples,n_atoms,n_dim=R.shape
    # R=np.reshape(R,(n_samples,n_atoms*n_dim))
    return np.array(R)

def extract_F_concat(dataset):

    F=dataset['F']
    n_samples,n_atoms,n_dim=F.shape
    F=np.reshape(F,(n_samples,n_atoms*n_dim))
    return np.array(F)

def mean_squared_error_sample_wise(x,y):
    err=(np.array(x)-np.array(y))**2
    return err.mean(axis=1)
