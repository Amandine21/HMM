import numpy as np
import math

#-------------------------------------------------------------------------
# Define the Global Variables
N = 3                                                                       # Hidden States
M = 4                                                                       # Observational Symbols
A_init = np.full((N, N), 1.0/N)                                             # Initialize the A matrix as uniform prior
B_init = np.full((N, M), 1.0/M)                                             # Initialize the B matrix as uniform prior
pi_init = np.zeros(N)                                                       # Initialize the hidden state vector
pi_init[0] = 1                                                              # Set the first hidden state to uniform prior

#-------------------------------------------------------------------------
# Computes the probability of observing the sequence up to time t with scaling
def forward(A, B, pi, O):
    N = A.shape[0]
    T = len(O)
    alpha = np.zeros((T, N))
    c = np.zeros(T)  # scaling factors

    # Base case
    alpha[0, :] = pi * B[:, O[0]]
    c[0] = 1.0 / np.sum(alpha[0, :])
    alpha[0, :] *= c[0]

    # Recursive case
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1, :] * A[:, j]) * B[j, O[t]]
        c[t] = 1.0 / np.sum(alpha[t, :])
        alpha[t, :] *= c[t]

    return alpha, c

#-------------------------------------------------------------------------
# Computes the probability of observing the sequence from time t+1 to the end with scaling
def backward(A, B, O, c):
    N = A.shape[0]
    T = len(O)
    beta = np.zeros((T, N))

    # Base case
    beta[T-1, :] = 1 * c[T-1]

    # Recursive case
    for t in range(T-2, -1, -1):
        for j in range(N):
            beta[t, j] = np.sum(A[j, :] * B[:, O[t+1]] * beta[t+1, :])
        beta[t, :] *= c[t]  # scale

    return beta

#-------------------------------------------------------------------------
# Baum-Welch algorithm
def baum_welch(O, N, M, A_init, B_init, pi_init, tol=1e-4, max_iter=100):
    A = A_init.copy()
    B = B_init.copy()
    pi = pi_init.copy()
    T = len(O)
    prev_log_likelihood = -np.inf
    O = np.array(O)

    for iteration in range(max_iter):
        alpha, c = forward(A, B, pi, O)
        beta = backward(A, B, O, c)

        gamma = np.zeros((T, N))
        xi = np.zeros((T-1, N, N))

        # Compute gamma
        for t in range(T):
            gamma[t] = alpha[t] * beta[t]
            gamma[t] /= np.sum(gamma[t])

        # Compute xi
        for t in range(T-1):
            denom = np.sum(alpha[t][:, None] * A * B[:, O[t+1]] * beta[t+1])
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = (alpha[t, i] * A[i, j] *
                                    B[j, O[t+1]] * beta[t+1, j]) / denom

        # Re-estimate pi
        pi = gamma[0]

        # Re-estimate A
        for i in range(N):
            denom = np.sum(gamma[:T-1, i])
            for j in range(N):
                numer = np.sum(xi[:, i, j])
                A[i, j] = numer / denom

        # Re-estimate B
        for i in range(N):
            denom = np.sum(gamma[:, i])
            for k in range(M):
                numer = np.sum(gamma[O == k, i])
                B[i, k] = numer / denom

        # Log-likelihood using scaling factors
        log_likelihood = -np.sum(np.log(c))

        if np.abs(log_likelihood - prev_log_likelihood) < tol:
            break
        prev_log_likelihood = log_likelihood

    return A, B, pi, log_likelihood

#-------------------------------------------------------------------------
def main():
    with open("hmm_c_N10000.in", "r") as f:
        O = list(map(int, f.read().split()))

    O = O[1:]
    A, B, pi, log_likelihood = baum_welch(O, N, M, A_init, B_init, pi_init, tol=1e-4, max_iter=50)

    print("\nTrained transition matrix A: ", A)
    print("\nTrained emission matrix B: ", B)
    print("\nTrained initial state pi: ", pi)
    print("\nFinal log-likelihood: ", log_likelihood)

# Run
main()

