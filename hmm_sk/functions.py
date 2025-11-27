import numpy as np
import math

#-------------------------------------------------------------------------
# Define the Global Variables

N = 3                                                                       # Hidden States
M = 4                                                                       # Observational Symbols

A_init = np.array([
    [1/3, 1/3, 1/3],
    [1/3, 1/3, 1/3],
    [1/3, 1/3, 1/3]

])

B_init = np.array([
    [1/4, 1/4, 1/4, 1/4],
    [1/4, 1/4, 1/4, 1/4],
    [1/4, 1/4, 1/4, 1/4]
])

pi_init = np.array([1/3, 1/3, 1/3])
     
#-------------------------------------------------------------------------
# Computes the probability of observing the sequence up to time t with scaling
def forward_algorithm(A, B, pi, observations):
    N = len(A)        # number of states
    T = len(observations)

    # Alpha table: alpha[t][i]
    alpha = []

    # ---- 1. INITIALIZATION ----
    first_obs = observations[0]
    alpha_t = []
    for i in range(N):
        value = pi[i] * B[i][first_obs]
        alpha_t.append(value)
    alpha.append(alpha_t)

    # ---- 2. INDUCTION ----
    for t in range(1, T):
        obs = observations[t]
        alpha_t = []

        # Compute alpha_t(i) for each state i
        for i in range(N):
            sum_prev = 0.0

            # sum over all previous states j
            for j in range(N):
                sum_prev += alpha[t - 1][j] * A[j][i]

            # multiply with emission probability
            value = sum_prev * B[i][obs]
            alpha_t.append(value)

        alpha.append(alpha_t)

    # ---- 3. TERMINATION ----
    # Probability of the whole observation sequence
    final_prob = sum(alpha[T - 1])

    return final_prob
#-------------------------------------------------------------------------
# Baum-Welch algorithm
def baum_welch(A, B, pi, observations, max_iters=100):
    N = len(A)          # number of states
    M = len(B[0])       # number of emission symbols
    T = len(observations)

    old_log_prob = float("-inf")

    for _ in range(max_iters):
        # ----- 1. FORWARD WITH SCALING -----
        alpha = [[0.0 for _ in range(N)] for _ in range(T)]
        c = [0.0 for _ in range(T)]  # scaling coefficients

        # Initialization alpha[0]
        c0 = 0.0
        first_obs = observations[0]
        for i in range(N):
            alpha[0][i] = pi[i] * B[i][first_obs]
            c0 += alpha[0][i]

        if c0 == 0.0:
            c0 = 1e-300
        c[0] = 1.0 / c0
        for i in range(N):
            alpha[0][i] *= c[0]

        # Induction alpha[t]
        for t in range(1, T):
            ct = 0.0
            ot = observations[t]

            for i in range(N):
                val = 0.0
                for j in range(N):
                    val += alpha[t - 1][j] * A[j][i]
                val *= B[i][ot]
                alpha[t][i] = val
                ct += val

            if ct == 0.0:
                ct = 1e-300
            c[t] = 1.0 / ct

            for i in range(N):
                alpha[t][i] *= c[t]

        # ----- 2. BACKWARD WITH SCALING -----
        beta = [[0.0 for _ in range(N)] for _ in range(T)]

        # Initialization beta[T-1]
        for i in range(N):
            beta[T - 1][i] = c[T - 1]

        # Induction beta[t]
        for t in range(T - 2, -1, -1):
            ot1 = observations[t + 1]
            for i in range(N):
                val = 0.0
                for j in range(N):
                    val += A[i][j] * B[j][ot1] * beta[t + 1][j]
                beta[t][i] = val * c[t]

        # ----- 3. GAMMA UND DIGAMMA BERECHNEN -----
        gamma = [[0.0 for _ in range(N)] for _ in range(T)]
        # digamma[t][i][j]
        digamma = [[[0.0 for _ in range(N)] for _ in range(N)] for _ in range(T - 1)]

        for t in range(T - 1):
            denom = 0.0
            ot1 = observations[t + 1]

            # Denominator für Normierung
            for i in range(N):
                for j in range(N):
                    denom += alpha[t][i] * A[i][j] * B[j][ot1] * beta[t + 1][j]

            if denom == 0.0:
                denom = 1e-300

            for i in range(N):
                gamma_sum_i = 0.0
                for j in range(N):
                    val = alpha[t][i] * A[i][j] * B[j][ot1] * beta[t + 1][j] / denom
                    digamma[t][i][j] = val
                    gamma_sum_i += val
                gamma[t][i] = gamma_sum_i

        # Letzte Gamma-Zeile (T-1) nur aus alpha
        denom_last = 0.0
        for i in range(N):
            denom_last += alpha[T - 1][i]
        if denom_last == 0.0:
            denom_last = 1e-300
        for i in range(N):
            gamma[T - 1][i] = alpha[T - 1][i] / denom_last

        # ----- 4. RE-ESTIMATE pi, A, B -----

        # pi
        for i in range(N):
            pi[i] = gamma[0][i]

        # A
        for i in range(N):
            denom = 0.0
            for t in range(T - 1):
                denom += gamma[t][i]

            for j in range(N):
                numer = 0.0
                for t in range(T - 1):
                    numer += digamma[t][i][j]

                if denom == 0.0:
                    A[i][j] = 0.0
                else:
                    A[i][j] = numer / denom

        # B
        for i in range(N):
            denom = 0.0
            for t in range(T):
                denom += gamma[t][i]

            for k in range(M):
                numer = 0.0
                for t in range(T):
                    if observations[t] == k:
                        numer += gamma[t][i]

                if denom == 0.0:
                    B[i][k] = 0.0
                else:
                    B[i][k] = numer / denom

        # ----- 5. LOG-LIKELIHOOD BERECHNEN UND KONVERGENZ PRÜFEN -----
        log_prob = 0.0
        for t in range(T):
            log_prob += math.log(c[t])
        log_prob = -log_prob

        # Abbruch, wenn sich Log-Likelihood nicht mehr verbessert
        if log_prob <= old_log_prob + 1e-6:
            break

        old_log_prob = log_prob

    return A, B, pi, log_prob