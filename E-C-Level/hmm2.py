def read_matrix():
    # Reads a matrix in the format:
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


def viterbi(A, B, pi, observations):
    # A: transition matrix (N x N)
    # B: emission matrix (N x M)
    # pi: initial distribution (length N)
    # observations: list of integers (0..M-1)

    N = len(A)              # number of states
    T = len(observations)   # length of observation sequence

    # delta[t][i] = best probability of a path ending in state i at time t
    delta = [[0.0 for _ in range(N)] for _ in range(T)]

    # backpointer[t][i] = argmax state index at time t-1 that led to i at time t
    backpointer = [[0 for _ in range(N)] for _ in range(T)]

    # ----- 1. INITIALIZATION -----
    first_obs = observations[0]
    for i in range(N):
        delta[0][i] = pi[i] * B[i][first_obs]
        backpointer[0][i] = 0  # no previous state at t=0, can keep 0 or -1

    # ----- 2. RECURSION -----
    for t in range(1, T):
        obs_t = observations[t]
        for i in range(N):
            best_val = -1.0
            best_state = 0

            # We choose the best predecessor j -> i
            for j in range(N):
                val = delta[t - 1][j] * A[j][i]
                if val > best_val:
                    best_val = val
                    best_state = j

            # Multiply with emission probability
            delta[t][i] = best_val * B[i][obs_t]
            backpointer[t][i] = best_state

    # ----- 3. TERMINATION -----
    # Find the best last state at time T-1
    best_last_state = 0
    best_prob = -1.0
    for i in range(N):
        if delta[T - 1][i] > best_prob:
            best_prob = delta[T - 1][i]
            best_last_state = i

    # ----- 4. BACKTRACKING -----
    path = [0 for _ in range(T)]
    path[T - 1] = best_last_state

    # Go backwards from T-1 to 0
    for t in range(T - 2, -1, -1):
        path[t] = backpointer[t + 1][path[t + 1]]

    return path


# ------------------ MAIN PROGRAM ------------------
def main():
    # Read HMM parameters
    A = read_matrix()          # transition matrix
    B = read_matrix()          # emission matrix
    pi_matrix = read_matrix()  # initial state distribution (1 x N)
    pi = pi_matrix[0]

    # Read observations: "T o1 o2 ... oT"
    parts = input().strip().split()
    T = int(parts[0])
    observations = list(map(int, parts[1:]))

    # Run Viterbi to get most likely state sequence
    state_sequence = viterbi(A, B, pi, observations)

    # Output: states as 0-based indices separated by spaces, no length
    print(" ".join(str(s) for s in state_sequence))


if __name__ == "__main__":
    main()
