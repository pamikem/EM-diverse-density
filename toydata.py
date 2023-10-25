import numpy as np



def gaussian_and_noise(n_pos=10, n_neg=10, n_samples_per_bag=200, seed=None):
    B = []
    l = np.zeros(n_pos+n_neg, dtype=int)
    np.random.seed(seed=seed)

    for i in range(n_pos):
        X_pos = np.random.randn(n_samples_per_bag//2, 2) * 0.25
        X_neg = np.random.rand(n_samples_per_bag//2, 2) * 2 - 1
        norms = np.linalg.norm(X_neg, axis=1)
        X_neg = X_neg[norms >= 0.5]
        B.append(np.r_[X_pos, X_neg])
        l[i] = 1

    for i in range(n_neg):
        X_neg = np.random.rand(n_samples_per_bag, 2) * 2 - 1
        norms = np.linalg.norm(X_neg, axis=1)
        X_neg = X_neg[norms >= 0.5]
        B.append(X_neg.copy())
        l[n_pos+i] = 0

    return B, l
    