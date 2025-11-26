def read_matrix():
    # Read the whole line and split it into strings
    parts = input().strip().split()

    # The first two numbers tell us rows and columns
    rows = int(parts[0])
    cols = int(parts[1])

    # The rest of the numbers are the actual matrix values
    numbers = list(map(float, parts[2:]))

    # Now we build the matrix row by row
    matrix = []
    index = 0

    for _ in range(rows):
        current_row = []
        for _ in range(cols):
            current_row.append(numbers[index])
            index += 1
        matrix.append(current_row)

    return matrix



# ---- Multiply row vector (1×N) with matrix (N×K) ----
def rowvec_matmul(row, mat):
    N = len(row)
    K = len(mat[0])
    result = [0.0] * K

    for j in range(K):
        s = 0.0
        for i in range(N):
            s += row[i] * mat[i][j]
        result[j] = s

    return result


# ---- MAIN ----
def main():
    # Read transition matrix A
    A = read_matrix()

    # Read emission matrix B
    B = read_matrix()

    # Read state distribution pi
    pi_matrix = read_matrix()
    pi = pi_matrix[0]   # row vector

    # Step 1: pi' = pi * A
    pi_next = rowvec_matmul(pi, A)

    # Step 2: observation distribution = pi' * B
    obs = rowvec_matmul(pi_next, B)

    # Output format: "1 M ..." (single row)
    print(1, len(obs), *obs)


if __name__ == "__main__":
    main()
