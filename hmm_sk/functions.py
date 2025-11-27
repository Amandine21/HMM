import math


     
#-------------------------------------------------------------------------
# Computes the probability of observing the sequence up to time t with scaling
def forward_algorithm(A, B, pi, observations):
    N = len(A)
    T = len(observations)

    # alpha_prev = alpha[0]
    alpha_prev = [0.0] * N
    first_obs = observations[0]
    for i in range(N):
        alpha_prev[i] = pi[i] * B[i][first_obs]

    # rekursiv
    for t in range(1, T):
        obs = observations[t]
        alpha_t = [0.0] * N

        for i in range(N):
            s = 0.0
            for j in range(N):
                s += alpha_prev[j] * A[j][i]
            alpha_t[i] = s * B[i][obs]

        alpha_prev = alpha_t

    return sum(alpha_prev)

#-------------------------------------------------------------------------
# Baum-Welch algorithm
def baum_welch(A, B, pi, observations, max_iters):
    N = len(A)
    M = len(B[0])
    T = len(observations)

    old_log_prob = float("-inf")

    for _ in range(max_iters):
        # ---------- 1. FORWARD MIT SCALING ----------
        alpha = [[0.0] * N for _ in range(T)]
        c = [0.0] * T

        # t = 0
        c0 = 0.0
        o0 = observations[0]
        for i in range(N):
            val = pi[i] * B[i][o0]
            alpha[0][i] = val
            c0 += val

        if c0 == 0.0:
            c0 = 1e-300
        c[0] = 1.0 / c0
        for i in range(N):
            alpha[0][i] *= c[0]

        # t >= 1
        for t in range(1, T):
            ct = 0.0
            ot = observations[t]
            alpha_t = alpha[t]
            alpha_tm1 = alpha[t-1]

            for i in range(N):
                s = 0.0
                # sum_j alpha[t-1][j] * A[j][i]
                for j in range(N):
                    s += alpha_tm1[j] * A[j][i]
                s *= B[i][ot]
                alpha_t[i] = s
                ct += s

            if ct == 0.0:
                ct = 1e-300
            c[t] = 1.0 / ct
            for i in range(N):
                alpha_t[i] *= c[t]

        # ---------- 2. BACKWARD MIT SCALING ----------
        beta = [[0.0] * N for _ in range(T)]
        # beta[T-1][i] = c[T-1]
        for i in range(N):
            beta[T-1][i] = c[T-1]

        for t in range(T-2, -1, -1):
            ot1 = observations[t+1]
            beta_t = beta[t]
            beta_tp1 = beta[t+1]

            for i in range(N):
                s = 0.0
                Ai = A[i]
                for j in range(N):
                    s += Ai[j] * B[j][ot1] * beta_tp1[j]
                beta_t[i] = s * c[t]

        # ---------- 3. GAMMA BERECHNEN & ZÄHLER/NENNER AUFBAUEN ----------

        # neue Parameter-Zähler initialisieren
        pi_new = [0.0] * N
        A_num = [[0.0] * N for _ in range(N)]
        A_den = [0.0] * N
        B_num = [[0.0] * M for _ in range(N)]
        B_den = [0.0] * N

        # t = 0 .. T-2: gamma(i,j) & gamma(i)
        for t in range(T-1):
            ot = observations[t]
            if t < T-1:
                ot1 = observations[t+1]

            # Nenner für gamma(i,j)
            denom = 0.0
            for i in range(N):
                for j in range(N):
                    denom += alpha[t][i] * A[i][j] * B[j][ot1] * beta[t+1][j]
            if denom == 0.0:
                denom = 1e-300

            for i in range(N):
                gamma_i = 0.0
                for j in range(N):
                    val = alpha[t][i] * A[i][j] * B[j][ot1] * beta[t+1][j] / denom
                    gamma_i += val

                    # A-Zähler
                    if t < T-1:
                        A_num[i][j] += val

                # A-Nenner: nur bis T-2
                if t < T-1:
                    A_den[i] += gamma_i

                # B: t zählt immer in den Nenner, Beobachtung ot in den Zähler
                B_den[i] += gamma_i
                B_num[i][ot] += gamma_i

                # π aus gamma_0
                if t == 0:
                    pi_new[i] = gamma_i

        # Speziell gamma(T-1): nur aus alpha
        denom_last = sum(alpha[T-1])
        if denom_last == 0.0:
            denom_last = 1e-300
        for i in range(N):
            gamma_last = alpha[T-1][i] / denom_last
            B_den[i] += gamma_last
            B_num[i][observations[T-1]] += gamma_last

        # ---------- 4. RE-ESTIMATE π, A, B ----------

        # pi
        for i in range(N):
            pi[i] = pi_new[i]

        # A
        for i in range(N):
            denom = A_den[i] if A_den[i] != 0.0 else 1e-300
            for j in range(N):
                A[i][j] = A_num[i][j] / denom

        # B
        for i in range(N):
            denom = B_den[i] if B_den[i] != 0.0 else 1e-300
            for k in range(M):
                B[i][k] = B_num[i][k] / denom

        # ---------- 5. LOG-LIKELIHOOD ----------
        log_prob = 0.0
        for t in range(T):
            log_prob += math.log(c[t])
        log_prob = -log_prob

        if log_prob <= old_log_prob + 1e-3:
            break
        old_log_prob = log_prob

    return A, B, pi, log_prob