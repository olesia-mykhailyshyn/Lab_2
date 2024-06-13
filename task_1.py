import numpy as np

matrix = np.mat("-1 1 -1;0 4 -2;0 4 -2")


def compute_eigenvalues_eigenvectors(matrix):
    print("Matrix:")
    print(matrix)
    print()

    evalue, evect = np.linalg.eig(matrix)

    print("Eigenvalues:")
    print(evalue)
    print()
    print("Eigenvectors:")
    print(evect)
    return evalue, evect


evalues, evectors = compute_eigenvalues_eigenvectors(matrix)

#shape = matrix.shape

print()
print("Checking A⋅v=λ⋅v for every eigenvalue and eigenvector:")
for i in range(len(evalues)):
    left = np.dot(matrix, evectors[:, i])
    right = evalues[i] * evectors[:, i]

    if np.allclose(left, right): #оскільки багато знаків після коми, треба перевірити максимально точно
        print("A⋅v=λ⋅v")
    else:
        print("There is a mistake: A⋅v!=λ⋅v")