import math


     
#-------------------------------------------------------------------------
# Computes the probability of observing the sequence up to time t with scaling

import math

#-------------------------------------------------------------------------
# Computes the log-probability of observing the sequence with scaling (numerical stability)
def forward_algorithm(A, B, pi, observations):
    N = len(A)        # number of states
    T = len(observations)
    
    # Alpha table: alpha[t][i] (scaled)
    alpha = [[0.0 for _ in range(N)] for _ in range(T)]
    c = [0.0 for _ in range(T)]  # scaling coefficients

    # ---- 1. INITIALIZATION (Scaled) ----
    c0 = 0.0
    first_obs = observations[0]
    for i in range(N):
        alpha[0][i] = pi[i] * B[i][first_obs]
        c0 += alpha[0][i]

    # Handle underflow/zero probability
    if c0 == 0.0:
        c0 = 1e-300
    c[0] = 1.0 / c0
    for i in range(N):
        alpha[0][i] *= c[0]

    # ---- 2. INDUCTION (Scaled) ----
    for t in range(1, T):
        ct = 0.0
        ot = observations[t]

        for i in range(N):
            val = 0.0
            # sum over all previous states j
            for j in range(N):
                val += alpha[t - 1][j] * A[j][i]
            
            # multiply with emission probability
            val *= B[i][ot]
            alpha[t][i] = val
            ct += val

        # Handle underflow/zero probability
        if ct == 0.0:
            ct = 1e-300
        c[t] = 1.0 / ct

        # Scale alpha[t]
        for i in range(N):
            alpha[t][i] *= c[t]

    # ---- 3. TERMINATION (Log-Probability) ----
    # log P(O|lambda) = - sum(log(c[t]))
    # This is a numerically stable value.
    log_prob = 0.0
    for t in range(T):
        # We must ensure c[t] is never zero for the log calculation
        log_prob += math.log(max(c[t], 1e-300))
    
    # The sum of logs of the scaling factors yields the log-likelihood
    # Note: Less negative log_prob is "better" (higher probability).
    return -log_prob

#-------------------------------------------------------------------------
# Baum-Welch algorithm
def baum_welch(A, B, pi, observations, max_iters):
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