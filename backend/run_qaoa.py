# Construction of the QUBO matrix

#%%
import json
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import LinearEqualityToPenalty

import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram
from qiskit import transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer  # Changed import to use qiskit_aer

from qiskit import qpy
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz

import time
from numpy.linalg import eigvalsh
from scipy.optimize import minimize

from qiskit import qpy
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info import hellinger_fidelity
from qiskit.visualization import plot_histogram, plot_state_city
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit_aer.noise import NoiseModel, depolarizing_error


#%%
# Parameters
capital = 1000 # total capital ($) to invest
nb_stocks = 5 # total number of stocks to consider

init_money_investment = [capital//nb_stocks]*nb_stocks
init_params_annealing = [np.pi, np.pi/2, np.pi, np.pi/2]
init_params = init_params_annealing + init_money_investment
print(f"Init parameters: {init_params}")




def generate_QUBO():
    """Code to generate the Pauli string from the QUBO formulation."""
    L = 1000 # power load to respect
    nb_units = 4
    p_max = 300
    p_min = 100

    qp = QuadraticProgram("optimisation")
    
    # Add variables
    for i in range(nb_units):
        qp.binary_var()

    # Inititalize the power to half-capacity
    power_units_init = [p_max/2] * nb_units
    qp.linear_constraint(linear=power_units_init, sense="==", rhs=L, name="lin_eq_load")
    print(qp.prettyprint())

    lineq2penalty = LinearEqualityToPenalty()
    qubo = lineq2penalty.convert(qp)
    print(qubo.prettyprint())
    
    pauli_operators, offset = qubo.to_ising()

    return pauli_operators, offset

#%%
pauli_op, offset = generate_QUBO()



#%%

def qaoa_circuit(sparse_pauli_op, p):
    """
    Create a QAOA circuit using a given cost Hamiltonian (SparsePauliOp).
    
    Args:
        sparse_pauli_op (SparsePauliOp): The cost Hamiltonian represented as a SparsePauliOp.
        p (int): Number of QAOA layers (depth of the circuit).

    Returns:
        QuantumCircuit: The QAOA quantum circuit.
    """
    num_qubits = sparse_pauli_op.num_qubits
    circuit = QuantumCircuit(num_qubits)

    # Create parameters for gamma and beta
    gamma_params = [Parameter(f'γ_{i}') for i in range(p)]
    beta_params = [Parameter(f'β_{i}') for i in range(p)]

    # Initialize qubits in the superposition state
    for qubit in range(num_qubits):
        circuit.h(qubit)

    # QAOA circuit with p layers
    for layer in range(p):
        # Apply the cost Hamiltonian (Cost Unitary)
        for pauli_term, coeff in zip(sparse_pauli_op.paulis, sparse_pauli_op.coeffs):
            z_indices = [i for i, pauli in enumerate(pauli_term.to_label()) if pauli == 'Z']

            # Apply a multi-controlled Z gate for each Z term in the Hamiltonian
            if len(z_indices) == 1:
                circuit.rz(2 * gamma_params[layer] * np.real(coeff), z_indices[0])
            elif len(z_indices) == 2:
                circuit.rzz(2 * gamma_params[layer] * np.real(coeff), z_indices[0], z_indices[1])

        # Apply the mixer Hamiltonian (Mixer Unitary)
        for qubit in range(num_qubits):
            circuit.rx(2 * beta_params[layer], qubit)

    return circuit


# Define a function to run the circuit with specific gamma and beta values
def run_qaoa(sparse_pauli_op, p, gamma_vals, beta_vals):
    """
    Runs the QAOA circuit with specific values of gamma and beta.
    
    Args:
        sparse_pauli_op (SparsePauliOp): The cost Hamiltonian represented as a SparsePauliOp.
        p (int): Number of QAOA layers (depth of the circuit).
        gamma_vals (list): List of gamma values for each layer.
        beta_vals (list): List of beta values for each layer.

    Returns:
        counts: The measurement counts from running the QAOA circuit.
    """
    # Generate the QAOA circuit
    qaoa_qc = qaoa_circuit(sparse_pauli_op, p)
    
    # Bind the parameters (gamma and beta)
    param_bindings = {}
    for i in range(p):
        param_bindings[f'γ_{i}'] = gamma_vals[i]
        param_bindings[f'β_{i}'] = beta_vals[i]
    
    # Bind parameters to the circuit
    qaoa_qc = qaoa_qc.assign_parameters(param_bindings)
    
    # Use Aer simulator to run the circuit
    simulator = Aer.get_backend('qasm_simulator')
    transpiled_qc = transpile(qaoa_qc, simulator)
    qobj = assemble(transpiled_qc, shots=1024)
    
    # Execute the circuit
    result = simulator.run(transpiled_qc).result()
    counts = result.get_counts()

    return counts

# Generate QAOA circuit with 1 layer (p=1)
qaoa_qc = qaoa_circuit(pauli_op, p=1)
print(qaoa_qc)

# Bind the parameters (gamma and beta)
param_bindings = {}
p = 1  # Number of layers (can increase if needed)
gamma_vals = [0.5]  # Example value for gamma
beta_vals = [0.3]   # Example value for beta
for i in range(p):
    param_bindings[f'γ_{i}'] = gamma_vals[i]
    param_bindings[f'β_{i}'] = beta_vals[i]

# Bind parameters to the circuit
qaoa_qc = qaoa_qc.assign_parameters(param_bindings)
print(qaoa_qc)

qaoa_qc.measure_all()


# Transpile for simulator
simulator = AerSimulator()
circ = transpile(qaoa_qc, simulator)

# Run and get counts
result = simulator.run(circ).result()
counts = result.get_counts(circ)
plot_histogram(counts, title='Sample')


# Get the string with the max number of counts
x_sol = max(counts) 


#%% ----- CLASSICAL OPTIMISATION ---------

def cost_func(params, ansatz, hamiltonian, estimator):
    """ Return estimate of energy from estimator.

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, hamiltonian, params)
    cost = estimator.run([pub]).result()[0].data.evs

    return cost

#%% -------------------- Minimisation routine --------------------
# TODO
# print("Minimisation routine.")
# start_time = time.time()

# # The simulation cannot be run on a circuit with classical measures
# result = minimize(cost_func_vqe,
#                   x0 = init_params,
#                   args=(isa_circuit, cost_hamiltonian, exact_estimator),
#                   method="COBYLA",
#                   options={'maxiter': 1000, 'disp': True})

# end_time = time.time()
# execution_time = end_time - start_time
# print(result)