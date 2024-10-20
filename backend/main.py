import QuantumRingsLib
from QuantumRingsLib import QuantumRegister, AncillaRegister, ClassicalRegister, QuantumCircuit
from QuantumRingsLib import QuantumRingsProvider
from QuantumRingsLib import job_monitor
from QuantumRingsLib import JobStatus
from skopt import gp_minimize
import numpy as np
import math
import os
import logging
logger = logging.getLogger(__name__)

import numpy as np
from qiskit import transpile, assemble


import json
from stock import Stock
import pandas as pd
import numpy as np

from skopt import gp_minimize

from qiskit_optimization.converters import InequalityToEquality
from qiskit_optimization.converters import IntegerToBinary
from qiskit_optimization.converters import LinearEqualityToPenalty
from qiskit_optimization import QuadraticProgram


provider = QuantumRingsProvider(token=os.environ.get('TOKEN_QUANTUMRINGS'), name=os.environ.get('ACCOUNT_QUANTUMRINGS'))
backend = provider.get_backend("scarlet_quantum_rings")
provider.active_account()


def ESG_scores(stocks):
    ESG = []
    for stock in stocks:
        ESG.append(stock.get_ESG_score())
    return ESG

def mean_returns(stocks):
    M =[]
    for stock in stocks:
        M.append(stock.get_annualized_returns())
    return M

def create_cov_matrix(stocks):
    # Create a DataFrame with the closing prices of the stocks
    df = pd.concat([stock.get_price_history()['Close'] for stock in stocks], axis=1)
    df.columns = [stock.Ticker for stock in stocks]
    
    # Calculate the percentage change
    returns = df.pct_change()
    
    # Calculate the covariance matrix
    cov_matrix = returns.cov()
    
    return cov_matrix

def Quant_qubo(w,mu_b,ESG_min,M,ESG_mat,Sigma,n):
    Q2dict={}
    for i in range(n):
        for j in range(i+1,n):
            if Sigma[i][j]!=0:
                Q2dict[(i,j)]=Sigma[i][j]*w[i]*w[j]
    Q1dict={}
    for i in range(n):
        if Sigma[i][i]!=0:
            Q1dict[i]=Sigma[i][i]*w[i]*w[i]
    LC1dict={}
    for i in range(n):
        LC1dict[i]=M[i]*w[i]
    LC2dict={}
    for i in range(n):
        LC2dict[i]=ESG_mat[i]*w[i]
    LC3dict={}
    for i in range(n):
        LC3dict[i]=1*w[i]

    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var("b"+str(i))
    qp.minimize(quadratic=Q2dict,linear=Q1dict)
    # qp.linear_constraint(linear=LC1dict, sense="GE", rhs=mu_b, name="Return constraint")
    # qp.linear_constraint(linear=LC2dict, sense="GE", rhs=ESG_min, name="ESG constraint")
    qp.linear_constraint(linear=LC3dict, sense="E", rhs=1, name="total investment")

    ineq2eq = InequalityToEquality()
    qp_eq = ineq2eq.convert(qp)
    int2bin = IntegerToBinary()
    qp_eq_bin = int2bin.convert(qp_eq)
    lineq2penalty = LinearEqualityToPenalty()
    qubo = lineq2penalty.convert(qp_eq_bin)

    return qubo

def Quant_qubo1(w,mu_b,ESG_min,M,ESG_mat,Sigma,n,Total_stocks):
    Q2dict={}
    for i in range(n):
        for j in range(i+1,n):
            if Sigma[i][j]!=0:
                Q2dict[(i,j)]=Sigma[i][j]*w[i]*w[j]
    Q1dict={}
    for i in range(n):
        if Sigma[i][i]!=0:
            Q1dict[i]=Sigma[i][i]*w[i]*w[i]
    LC1dict={}
    for i in range(n):
        LC1dict[i]=M[i]*w[i]
    LC2dict={}
    for i in range(n):
        LC2dict[i]=ESG_mat[i]*w[i]
    LC3dict={}
    for i in range(n):
        LC3dict[i]=1*w[i]

    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var("b"+str(i))
    qp.minimize(quadratic=Q2dict,linear=Q1dict)
    # qp.linear_constraint(linear=LC1dict, sense="GE", rhs=mu_b, name="Return constraint")
    # qp.linear_constraint(linear=LC2dict, sense="GE", rhs=ESG_min, name="ESG constraint")
    qp.linear_constraint(linear=LC3dict, sense="E", rhs=Total_stocks, name="total investment")

    ineq2eq = InequalityToEquality()
    qp_eq = ineq2eq.convert(qp)
    int2bin = IntegerToBinary()
    qp_eq_bin = int2bin.convert(qp_eq)
    lineq2penalty = LinearEqualityToPenalty()
    qubo = lineq2penalty.convert(qp_eq_bin)

    return qubo
def Class_qubo(b):
    Q2dict={}
    for i in range(n):
        for j in range(i+1,n):
            if Sigma[i][j]!=0:
                Q2dict[(i,j)]=Sigma[i][j]*b[i]*b[j]
    Q1dict={}
    for i in range(n):
        if Sigma[i][i]!=0:
            Q1dict[i]=Sigma[i][i]*b[i]*b[i]
    LC1dict={}
    for i in range(n):
        LC1dict[i]=M[i]*b[i]
    LC2dict={}
    for i in range(n):
        LC2dict[i]=ESG_mat[i]*b[i]
    LC3dict={}
    for i in range(n):
        LC3dict[i]=1*b[i]

    qp = QuadraticProgram()
    for i in range(n):
        # qp.integer_var(lowerbound=0, upperbound=capital, name="w"+str(i))
        qp.continuous_var(lowerbound=0, upperbound=1, name="w"+str(i))
        
        # qp.continuous_var("w"+str(i))
        # binary_var("w"+str(i))
    qp.minimize(quadratic=Q2dict,linear=Q1dict)

    # qp.linear_constraint(linear=LC1dict, sense="GE", rhs=mu_b, name="Return constraint")
    # qp.linear_constraint(linear=LC2dict, sense="GE", rhs=ESG_min, name="ESG constraint")
    qp.linear_constraint(linear=LC3dict, sense="E", rhs=1, name="total investment")

    ineq2eq = InequalityToEquality()
    qp_eq = ineq2eq.convert(qp)
    # int2bin = IntegerToBinary()
    # qp_eq_bin = int2bin.convert(qp_eq)
    qp_eq_bin = qp_eq
    lineq2penalty = LinearEqualityToPenalty()
    qubo = lineq2penalty.convert(qp_eq_bin)

    return qubo

def Operator_UB(graph, beta,qc, q, n_qubits):
    for i in range(n_qubits): qc.rx(2*beta, q[i])

# define the operator U(C,gamma)
def Operator_UC(graph, gamma, qc, q, n_qubits):
    CoeffMatrix=graph
    for i in range(n_qubits):
        for j in range(i+1,n_qubits):
            if CoeffMatrix[i,j]!=0:
                qc.cx(q[i], q[j])
                qc.rz(2*gamma*CoeffMatrix[i,j], q[j])
                qc.cx(q[i], q[j])
    for i in range(n_qubits):
        qc.rz(2*gamma*(-CoeffMatrix[i,i]+sum(CoeffMatrix[i])), q[i])
    # for i in range(len(PauliList)):
    #     pauli=str(PauliList[i])
    #     coeff=Paulicoeff[i]
    #     indices=[]
    #     for j in range(len(pauli)):
    #         if pauli[j]=='Z':
    #             indices.append(j)
    #     if len(indices)==1:
    #         print(indices)
    #         qc.rz(2*gamma*coeff, q[indices[0]])
    #     elif len(indices)==2:
    #         print(indices)
    #         qc.cx(q[indices[0]], q[indices[1]])
    #         qc.rz(2*gamma*coeff, q[indices[1]])
    #         qc.cz(q[indices[0]], q[indices[1]])

# a helper routine that computes the total weight of the cuts
def WeightOfCuts(bitstring,graph):
    bitstring=[int(i) for i in bitstring]
    totalWeight=bitstring@graph[1]@bitstring
    totalWeight=totalWeight+graph[2]
    return totalWeight

def jobCallback(job_id, state, job):
    #print("Job Status: ", state)
    pass

def jobCallback(job_id, state, job):
    #print("Job Status: ", state)
    pass

# Builds the QAOA state.
def qaoaState( x, graph, p, n_qubits, expectationValue = True, shots=1024):
    gammas = []
    betas = []
    # setup the gamma and beta array
    for i in range(len(x)//2):
        gammas.append(x[i*2])
        betas.append(x[(i*2)+1])
    # Create the quantum circuit
    q = QuantumRegister(n_qubits)
    c = ClassicalRegister(n_qubits)
    qc = QuantumCircuit(q, c)

    # First set the qubits in an equal superposition state
    for i in range(n_qubits):
        qc.h(q[i])

    # Apply the gamma and beta operators in repetition
    for i in range(p):
        # Apply U(C,gamma)
        Operator_UC(graph[0], gammas[i], qc, q, n_qubits)

        # Apply U(B, beta)
        Operator_UB(graph[0], betas[i],qc, q, n_qubits)

    # Measure the qubits
    for i in range(n_qubits):
        qc.measure(q[i], c[i])

    # Execute the circuit now
    job = backend.run(qc, shots)
    job.wait_for_final_state(0, 5, jobCallback)
    counts = job.result().get_counts()

    # decide what to return
    if ( True == expectationValue ):
        # Calculate the expectation value of the measurement.
        expectation = 0
        for bitstring in counts:
            probability = float(counts[bitstring]) / shots
            expectation += WeightOfCuts(bitstring,graph) * probability
        return( expectation )
    else:
        # Just return the counts return(counts)
        return(counts)
def run(input_data,solver_params,extra_args):
    stock_data=input_data
    stocks = [Stock(stock) for stock in stock_data.values()]
    #inputs
    num_stocks=5
    num_features = 4
    mu_b=1
    ESG_min=0.1
    M=mean_returns(stocks)[:num_features]
    ESG_mat = np.array(ESG_scores(stocks))[:num_features,0]
    Sigma=np.array(create_cov_matrix(stocks))[:num_features,:num_features]
    Total_Stocks=5
    
    ESG_mat = np.repeat(ESG_mat, num_stocks, axis=0)
    M = np.repeat(M, num_stocks, axis=0)
    Sigma = np.repeat(Sigma, num_stocks, axis=0)
    Sigma = np.repeat(Sigma, num_stocks, axis=1)

    #circuit parameters
    p = 3                # Number of circuit layers
    n_qubits = num_features*num_stocks        # Number of qubits
    n_calls = 100        # Optimization cycles
    n_random_starts = 2  # See scikit documentation
    dimensions = [] # Search space, place holder

    w=np.ones(n_qubits)
    logger.info("Starting solver execution...")

    # Generate QUBO.
    
    qubo=Quant_qubo1(w,mu_b,ESG_min,M,ESG_mat,Sigma,n_qubits,Total_Stocks)
    print(qubo.prettyprint())
    pauli_op = qubo.to_ising()[0]
    PauliList=pauli_op._pauli_list
    Paulicoeff=pauli_op._coeffs
    CoeffMatrix=np.zeros((n_qubits,n_qubits))
    for i in range(len(PauliList)):
        pauli=str(PauliList[i])
        coeff=Paulicoeff[i]
        indices=[]
        for j in range(len(pauli)):
            if pauli[j]=='Z':
                indices.append(j)
        if len(indices)==1:
            CoeffMatrix[indices[0],indices[0]]=coeff
        elif len(indices)==2:
            CoeffMatrix[indices[0],indices[1]]=coeff
            CoeffMatrix[indices[1],indices[0]]=coeff
    
    Q=np.zeros((n_qubits,n_qubits))
    for key, value in qubo._objective._quadratic._coefficients.items():
        Q[key[0]][key[1]]=value
        Q[key[1]][key[0]]=value
    for key, value in qubo._objective._linear._coefficients.items():
        Q[key[1]][key[1]]+=value
    graph = [CoeffMatrix,Q,qubo._objective._constant]
    
    # Construct the search space depending upon the circuit layers
    for i in range(p):
        dimensions.append((0,2*np.pi))
        dimensions.append((0,np.pi))
    d = tuple(dimensions)
    
    # setup the optimization function, as its negative
    f = lambda x: qaoaState(x, graph, p, n_qubits)
    
    # Use the Bayesian optimization using Gaussian Processes from Scikit optimizer
    # to maximize the cost function (by minimizing its negative)
    res = gp_minimize(func=f,
            dimensions = d,
            n_calls=n_calls,
            n_random_starts=n_random_starts)
    
    # Fetch the optimal gamma and beta values
    x = res.x
    # Execute the qaoa state with the optimal gamma and beta values
    counts = qaoaState(x, graph, p, n_qubits, False)

    logger.info("End of solver execution.")
    print("done")

    Total_Stocks=5
    num_stocks=5
    max_counts=0
    opt_string=0
    for string in counts:
        if sum([int(i) for i in string]) ==Total_Stocks:
            if counts[string]>max_counts:
                opt_string=string
                max_counts=counts[string]

    Total_Money=1000
    for i in range(len(opt_string)//num_stocks):
        print(opt_string[i*num_stocks:i*num_stocks+num_stocks])
        stt=[int(k) for k in opt_string[i*num_stocks:i*num_stocks+num_stocks]]
        print("In stock:",i," invest", sum(stt)*Total_Money/Total_Stocks)
    return counts
    
