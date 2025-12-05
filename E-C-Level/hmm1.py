def read_matrix():
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


# Forward algorithm for HMM1
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
    print(f"First alpha = {alpha_t}")

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
        print(f"Alpha {t} = {alpha_t}")

    # ---- 3. TERMINATION ----
    # Probability of the whole observation sequence
    #print(f"Alpha list = {alpha}")
    final_prob = sum(alpha[T - 1])

    return final_prob


# ------------------ MAIN PROGRAM ------------------
def main():
    A = read_matrix()         # Transition matrix
    B = read_matrix()         # Emission matrix
    pi_matrix = read_matrix() # Initial distribution
    pi = pi_matrix[0]

    # Read observations
    parts = input().strip().split()
    T = int(parts[0])
    observations = list(map(int, parts[1:]))

    print(f"observations = {observations}")

    # Run forward algorithm
    result = forward_algorithm(A, B, pi, observations)

    # Print final probability
    print(result)


if __name__ == "__main__":
    main()
