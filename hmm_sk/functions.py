# functions.py
import math

# -------------------------------------------------------------------------
# Forward-Algorithmus mit Scaling, gibt log P(O | λ) zurück.
# Optimiert für das Guessing (nur 1D-alpha, keine große Matrix).
# -------------------------------------------------------------------------
def forward_algorithm(A, B, pi, observations):
    N = len(A)
    T = len(observations)

    # alpha[t] only stored for one step (memory optimal)
    alpha = [0.0] * N
    c = [0.0] * T

    # --- t = 0 ---
    o0 = observations[0]
    scale = 0.0
    for i in range(N):
        alpha[i] = pi[i] * B[i][o0]
        scale += alpha[i]

    if scale < 1e-300:
        scale = 1e-300

    c[0] = 1.0 / scale
    for i in range(N):
        alpha[i] *= c[0]

    # --- t = 1 ... T-1 ---
    for t in range(1, T):
        ot = observations[t]
        new_alpha = [0.0] * N
        scale = 0.0

        for i in range(N):
            s = 0.0
            for j in range(N):
                s += alpha[j] * A[j][i]
            s *= B[i][ot]
            new_alpha[i] = s
            scale += s

        if scale < 1e-300:
            scale = 1e-300

        c[t] = 1.0 / scale
        for i in range(N):
            new_alpha[i] *= c[t]

        alpha = new_alpha

    # --- final log-likelihood ---
    log_prob = 0.0
    for t in range(T):
        log_prob += math.log(c[t])

    return -log_prob



# -------------------------------------------------------------------------
# Baum-Welch mit Scaling (klassisch), leicht bereinigt.
# Nutzt 2D alpha/beta, weil wir Gamma brauchen.
# -------------------------------------------------------------------------
def baum_welch(A, B, pi, observations, max_iters):
    N = len(A)          # number of states
    M = len(B[0])       # number of emission symbols
    T = len(observations)

    old_log_prob = float("-inf")

    for _ in range(max_iters):
        # ----- 1. FORWARD MIT SCALING -----
        alpha = [[0.0 for _ in range(N)] for _ in range(T)]
        c = [0.0 for _ in range(T)]

        # t = 0
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

        # t = 1..T-1
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

        # ----- 2. BACKWARD MIT SCALING -----
        beta = [[0.0 for _ in range(N)] for _ in range(T)]

        # t = T-1
        for i in range(N):
            beta[T - 1][i] = c[T - 1]

        # t = T-2..0
        for t in range(T - 2, -1, -1):
            ot1 = observations[t + 1]
            for i in range(N):
                val = 0.0
                for j in range(N):
                    val += A[i][j] * B[j][ot1] * beta[t + 1][j]
                beta[t][i] = val * c[t]

        # ----- 3. GAMMA UND DIGAMMA -----
        gamma = [[0.0 for _ in range(N)] for _ in range(T)]
        digamma = [[[0.0 for _ in range(N)] for _ in range(N)] for _ in range(T - 1)]

        for t in range(T - 1):
            denom = 0.0
            ot1 = observations[t + 1]

            for i in range(N):
                for j in range(N):
                    denom += alpha[t][i] * A[i][j] * B[j][ot1] * beta[t + 1][j]

            if denom == 0.0:
                denom = 1e-300

            for i in range(N):
                gamma_sum_i = 0.0
                for j in range(N):
                    val = (alpha[t][i] * A[i][j] *
                           B[j][ot1] * beta[t + 1][j] / denom)
                    digamma[t][i][j] = val
                    gamma_sum_i += val
                gamma[t][i] = gamma_sum_i

        # letztes Gamma nur aus alpha
        denom_last = sum(alpha[T - 1])
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

                A[i][j] = 0.0 if denom == 0.0 else numer / denom

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

                B[i][k] = 0.0 if denom == 0.0 else numer / denom

        # ----- 5. LOG-LIKELIHOOD + KONVERGENZ -----
        log_prob = 0.0
        for t in range(T):
            log_prob += math.log(c[t])
        log_prob = -log_prob

        # Toleranz etwas lockerer, damit wir früher abbrechen
        if log_prob <= old_log_prob + 1e-3:
            break

        old_log_prob = log_prob

    return A, B, pi, log_prob
