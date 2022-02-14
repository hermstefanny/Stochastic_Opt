import numpy as np
import matplotlib.pyplot as plt


def estimate(X, W, K, Z_b):
    A = X @ W
    sigmoid_Z = 1 / (1 + np.exp(-A))
    Z = np.append(Z_b, sigmoid_Z, axis=1)
    Y_hat = Z @ K
    return Y_hat, Z


if __name__ == '__main__':
    obs = 8
    X_0 = np.ones(obs)
    X_1 = np.array([0.003, 0.00275, 0.0025, 0.00225, 0.002, 0.00175,
                    0.0015, 0.00125])*500

    Y = np.transpose(np.array([[885.14008, 847.9070105586089, 847.9070105586089,
                                742.7306349753593, 672.117239375, 635.2425071858478,
                                849.19106, 2773.527600223999]])/1000)
    miu = 0.01
    H = 2
    N = 3

    np.random.seed(2022)
    X = np.transpose(np.stack([X_0, X_1]))
    W = 1 - np.random.rand(H, N)
    Z_bias = np.ones((obs, 1))

    K = 1 - np.random.rand(N + 1, 1)

    SSE_old = 100
    error = 10
    iterations = 0
    iterations_max = 100000
    tol = 0.001
    
    while SSE > tol and iterations < iterations_max:
        Y_hat, Z = estimate(X, W, K, Z_bias)
        for i in range(N):
            for j in range(H):
                W[j, i] = W[j, i] + np.asscalar(2 * miu * sum((Y - Y_hat) * K[i + 1, 0] * Z[:, i + 1:i + 2] *
                                                              (np.ones((obs, 1)) - Z[:, i + 1:i + 2]) *
                                                              X[:, j:j + 1]))

        for k in range(N + 1):
            K[k, 0] = K[k, 0] + np.asscalar(2 * miu * sum((Y - Y_hat) * Z[:, k:k + 1]))


        iterations = iterations + 1
        miu = miu * (500 / (1000 + iterations))
        error = abs(np.sum(Y - Y_hat))
        SSE = np.sum(np.square(Y - Y_hat), axis=0)


        if iterations % 1000 == 0:
            print(f"SSE {SSE} and Error {error}")
            print(f'Iterations {iterations}')

    Y_hat, Z = estimate(X, W, K, Z_bias)
    New_Y = Y_hat*1000
    print(f"Results: W\n {W} \n and K\n {K}")
    print(f"Which give this estimation: {New_Y}")

    points = 1000
    x_new_1 = np.linspace(0, 0.0035, points) * 500
    x_new_0 = np.ones(points)
    X_new = np.transpose(np.stack([x_new_0, x_new_1]))

    Z_b_new = np.ones((points, 1))
    Y_invented, Z_invented = estimate(X_new, W, K, Z_b_new)

    value_min = np.min(Y_invented) * 1000
    pos_v_min = np.argmin(Y_invented)
    prob_min = (x_new_1[pos_v_min])/500

    print(f"The minimum in the new set is : {value_min}")
    print(f"The probability minimum is : {prob_min}")

    plt.scatter(X_1/500, Y*1000)
    plt.scatter(X_1/500, New_Y)
    plt.plot(x_new_1/500, Y_invented*1000, c='g')
    plt.yscale('log')
    plt.show()