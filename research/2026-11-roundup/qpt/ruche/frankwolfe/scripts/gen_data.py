import sys, types
sys.modules["mpi4py"] = types.SimpleNamespace(MPI=None)

import sys
import os 
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from qiskit.quantum_info import process_fidelity, Chi, Operator, Kraus
from qutomo import states, projectors, measurements, methods

sys.path.append('../qutomo') 

""" 
settings that can be modified
"""

n = 3 #no of qubits
np.random.seed(0)
_eps = 0.05 #Gaussain noise level to be added
SAVE_NAME = f'jax_arr_gt_xi_{_eps}.h5' #exact path where data will be saved
backend    = 'qasm_simulator' #qiskit simulator
num_shots  = 1000 #no of shots to generate outcomes

#total no of (X_ij, y_ij) pairs:
(4**n) * (2**n)

#dimensions of Chi:
f"{(2 ** n)**2} x {(2 ** n)**2}"

#Building global metadata
import itertools

rho_in_labels = [''.join(p) for p in itertools.product(['0', '1', '+', 'i'], repeat=n)] #4^n
meas_setting_labels = [''.join(p) for p in itertools.product(['X', 'Y', 'Z'], repeat=n)] #3^n settings per rho_in

""" 
obtain orthonormal Pauli basis {\tilde{A}_n}
"""
A_basis, A_labels = projectors.get_pauli_orthonormal_basis(n)
A_basis = np.array(A_basis)
A_basis = A_basis[np.argsort(A_labels)]

global_setting_data = {}
for setting in meas_setting_labels:
    SETTING_KEY = f'setting_{setting}'
    # Setting-wide data (Independent of outcome)
    global_setting_data[SETTING_KEY] = {
        'P_matrix': projectors.build_projector(setting), # e.g., the actual X@Y matrix | Pauli matrix / projector P_i corresponding to a circuit setting
        'outcomes': {}
    }
    
    # Outcome-specific data
    v_ij_map = projectors.build_projection_vector_dict(setting)
    for outcome, vector in v_ij_map.items():
        OUTCOME_KEY = f'outcome_{outcome}'
        global_setting_data[SETTING_KEY]['outcomes'][OUTCOME_KEY] = {
            'E_j': np.outer(vector, vector.conj().T) # The specific |01><01|
        }

""" 
Each of the 3^n settings has:
"P_matrix" -> the Pauli matrix corresponding to it
"outcomes" -> dictionary of outcomes with { "outcome_{val}": {"E_j" : matrix} }
"""
global_setting_data.keys()

qpt_results = {
    'metadata': {'A_basis': A_basis, 'A_labels': A_labels},
    'experiments': {} 
}

for p_label in rho_in_labels:
    state = states.PauliPrepState(n, p_label)
    state.create_circuit() # Ensure circuit is built

    rho_in = state.get_state_matrix() #getting state matrix
    RHO_KEY = f'rho_{p_label}'
    
    exec_data = state.execute_measurement_circuits(meas_setting_labels, backend=backend, num_shots=num_shots) #running shots to get measurement data
    
    # Store this rho_in's data
    exp_entry = {
        'rho_in': rho_in,
        'settings': {} 
    }

    for data_dict_map_per_setting in exec_data:
        setting = data_dict_map_per_setting['label']
        SETTING_KEY = f'setting_{setting}'
        counts = data_dict_map_per_setting['count_dict']
        meas_obj = measurements.Measurement(setting, counts)
        
        # SETTING LEVEL
        # f_i: The Pauli Expectation Value (e.g., <XY>)
        f_i = meas_obj.get_pauli_correlation_measurement()[setting]
        
        exp_entry['settings'][SETTING_KEY] = {
            'f_i_expectation': f_i,
            'outcomes': {}
        }

        f_ij_vec = meas_obj.get_pauli_basis_measurement()[setting]

        # OUTCOME LEVEL

        #the outcome keys (2^n) are common to all settings
        for outcome_key in global_setting_data[SETTING_KEY]['outcomes'].keys():
            OUTCOME_KEY = outcome_key
            #f_ij: The specific probability
            _outcome = OUTCOME_KEY.split('_')[1]
            f_ij = f_ij_vec[_outcome]
            
            #D_ij: Computed using the outcome-specific E_j
            E_j = global_setting_data[SETTING_KEY]['outcomes'][OUTCOME_KEY]['E_j']

            #get D_ij matrix is custom built function as per paper
            D_ij = projectors.get_Dij_matrix(
                rho_in=rho_in, 
                E_j=E_j, 
                A_basis= qpt_results['metadata']['A_basis'], 
                labels= qpt_results['metadata']['A_labels']
            )
            
            exp_entry['settings'][SETTING_KEY]['outcomes'][OUTCOME_KEY] = {
                'f_ij_probability': f_ij,
                'D_ij_matrix': D_ij,
            }
            
    qpt_results['experiments'][RHO_KEY] = exp_entry

#4^n input states
rho_mapping = {r:RHO_KEY for r, RHO_KEY in enumerate(sorted(qpt_results['experiments'])) }
#3^n settings
i_mapping = {i:SETTING_KEY for i, SETTING_KEY in enumerate(sorted(global_setting_data.keys()))}
#2^n outcomes
j_mapping = {j:OUTCOME_KEY for j, OUTCOME_KEY in enumerate(sorted(global_setting_data[SETTING_KEY]['outcomes']))}

d = 2**n

#PRE-PROCESSING
# Flatten all D_ij and f_ij into lists to avoid nested dict lookups in every iteration
all_D = []
all_f = []

for p_label, exp in qpt_results['experiments'].items():
    for setting, set_data in exp['settings'].items():
        for outcome, out_data in set_data['outcomes'].items():
            all_D.append(out_data['D_ij_matrix'])
            all_f.append(out_data['f_ij_probability'])

# Convert to numpy arrays for speed
D_tensors = np.array(all_D) # Shape: (Total_measurements, d^2, d^2)
f_vector = np.array(all_f)  # Shape: (Total_measurements,)

## To use the ground truth

from qiskit.quantum_info import Operator, Chi

def haar_unitary(n):
    """ 
    From:
    @misc{mezzadri2007generaterandommatricesclassical,
        title={How to generate random matrices from the classical compact groups}, 
        author={Francesco Mezzadri},
        year={2007},
        eprint={math-ph/0609050},
        archivePrefix={arXiv},
        primaryClass={math-ph},
        url={https://arxiv.org/abs/math-ph/0609050}, 
    }
    """
    Z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    phase = np.diag(R) / np.abs(np.diag(R))
    Q = Q * phase[np.newaxis, :]
    return Q

# 1. Create the physical Unitary
H = haar_unitary(d) 

# 2. Vectorize the Unitary in the way your basis expects
# Mathematically, Chi_mn = Tr(A_m.dag @ H) * Tr(A_n @ H.dag) / d
# But a cleaner way is to find the components 'c' such that H = sum(c_i * A_i)

# First, get the coefficients of U in your Pauli Basis
# c_i = Tr(A_i.dag @ _x) / d
_c = np.array([np.trace(A.conj().T @ H) for A in A_basis]) # / d

# 3. Construct Chi_star from these coefficients
# Chi_ij = c_i * conj(c_j)
Chi_star = np.outer(_c, np.conj(_c))

# 4. Verify physical trace (Tr(Chi) should be d)
print(f"Trace of Chi_star: {np.trace(Chi_star):.4f} (Should be {d})")

Chi_star_channel = Chi(data = Chi_star)

f_vector_gt = np.zeros_like(f_vector, dtype = f_vector.dtype)
for i, D in enumerate(D_tensors):
    f_vector_gt[i] = np.vdot(D, Chi_star).real + _eps * np.random.standard_normal() 

f_vector = f_vector_gt.copy()
del f_vector_gt

print(f"Chi_star has NaNs: {np.isnan(Chi_star).any()}")
print(f"Chi_star has Infs: {np.isinf(Chi_star).any()}")

# Check the D_tensors list
d_nan = any(np.isnan(d).any() for d in D_tensors)
d_inf = any(np.isinf(d).any() for d in D_tensors)
print(f"D_tensors contains NaNs: {d_nan}")
print(f"D_tensors contains Infs: {d_inf}")

m = f_vector.shape[0]

B_tensors = np.zeros((d**2, d**2, d, d), dtype = np.complex128)
for _i in range(d**2):
    for _j in range(d**2):
        B_tensors[_i, _j, :, :] = A_basis[_j].conj().T @ A_basis[_i]

assert m == 24**n #no of observations in total is of this form

D_tensors.shape, f_vector.shape, B_tensors.shape, A_basis.shape

import h5py
import jax.numpy as jnp

f_jax_vector = jnp.array(f_vector)
D_jax_tensors = jnp.array(D_tensors)
A_jax_basis = jnp.array(A_basis)
B_jax_tensors = jnp.array(B_tensors)
Chi_star_tensor = jnp.array(Chi_star)

with h5py.File(SAVE_NAME, 'w') as f:
    f.create_dataset('f_jax_vector',  data=f_jax_vector)
    f.create_dataset('D_jax_tensors', data=D_jax_tensors)
    f.create_dataset('A_jax_basis',   data=A_jax_basis)
    f.create_dataset('B_jax_tensors', data=B_jax_tensors)
    f.create_dataset('Chi_star_tensor', data=Chi_star_tensor)