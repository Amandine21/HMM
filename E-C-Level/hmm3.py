import math
import matplotlib.pyplot as plt


def read_matrix():
    # Reads a matrix:
    # rows cols v11 v12 ... v1c v21 ... etc.
    parts = input().strip().split()

    rows = int(parts[0])
    cols = int(parts[1])

    numbers = list(map(float, parts[2:]))

    matrix = []
    index = 0

    for _ in range(rows):
        current_row = []
        for _ in range(cols):
            current_row.append(numbers[index])
            index += 1
        matrix.append(current_row)

    return matrix


def avg_abs_diff_vector(v_old, v_new):
    n = len(v_old)
    return sum(abs(v_old[i] - v_new[i]) for i in range(n)) / n


def avg_abs_diff_matrix(M_old, M_new):
    rows = len(M_old)
    cols = len(M_old[0])
    total = 0.0
    for i in range(rows):
        for j in range(cols):
            total += abs(M_old[i][j] - M_new[i][j])
    return total / (rows * cols)


def baum_welch(A, B, pi, observations, max_iters=100):
    N = len(A)          # number of states
    M = len(B[0])       # number of emission symbols
    T = len(observations)

    old_log_prob = float("-inf")

    # histories for convergence plot
    delta_pi_hist = []
    delta_A_hist = []
    delta_B_hist = []

    for iter in range(max_iters):
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

        # store old parameters for convergence plot
        old_pi = pi[:]                    # copy vector
        old_A = [row[:] for row in A]     # deep copy
        old_B = [row[:] for row in B]

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

        # ----- compute parameter changes for this iteration -----
        delta_pi = avg_abs_diff_vector(old_pi, pi)
        delta_A = avg_abs_diff_matrix(old_A, A)
        delta_B = avg_abs_diff_matrix(old_B, B)

        delta_pi_hist.append(delta_pi)
        delta_A_hist.append(delta_A)
        delta_B_hist.append(delta_B)

        # ----- 5. LOG-LIKELIHOOD BERECHNEN UND KONVERGENZ PRÜFEN -----
        log_prob = 0.0
        for t in range(T):
            log_prob += math.log(c[t])
        log_prob = -log_prob

        # Abbruch, wenn sich Log-Likelihood nicht mehr verbessert
        if log_prob <= old_log_prob + 1e-2:
            break

        #print(f"Iteration {iter}: \nA = {A}, \nB = {B}, \npi = {pi}")

        old_log_prob = log_prob

    # return histories as well for plotting
    return A, B, delta_pi_hist, delta_A_hist, delta_B_hist


def print_matrix(mat):
    rows = len(mat)
    cols = len(mat[0])

    values = []
    for i in range(rows):
        for j in range(cols):
            # 6 Nachkommastellen sind meist ausreichend
            values.append("{:.6f}".format(mat[i][j]))

    print(rows, cols, " ".join(values))


# ------------------ MAIN PROGRAM ------------------
def main():
    # Read observations: "T o1 o2 ... oT"
    parts = input().strip().split()
    T = int(parts[0])
    observations = list(map(int, parts[1:]))

    A = [
        [0.7, 0.05, 0.25],
        [0.13, 0.72, 0.15],
        [0.14, 0.26, 0.6]
    ]

    B = [
        [0.7, 0.19, 0.1, 0.01],
        [0.09, 0.43, 0.32, 0.16],
        [0.05, 0.17, 0.19, 0.59]
    ]

    pi = [0.3, 0.2, 0.5]

    # Train HMM with Baum-Welch
    A_est, B_est, d_pi, d_A, d_B = baum_welch(A, B, pi, observations)

    # Output only A and B (for the assignment)
    print_matrix(A_est)
    print_matrix(B_est)

    # ---- plot convergence (for your report / local run) ----
    iters = range(1, len(d_pi) + 1)

    plt.figure()
    plt.plot(iters, d_pi, label="Δπ (avg abs diff)")
    plt.plot(iters, d_A, label="ΔA (avg abs diff)")
    plt.plot(iters, d_B, label="ΔB (avg abs diff)")
    plt.xlabel("Iteration")
    plt.ylabel("Average parameter change")
    plt.title("Convergence of Baum–Welch parameters")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
