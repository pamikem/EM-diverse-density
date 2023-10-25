import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans




class EM_DD(BaseEstimator):

    def __init__(self, random_state=None, n_init_bags=1):
        self.random_state = random_state
        self.n_init_bags = n_init_bags

    def fit(self, B, l):
        np.random.seed(self.random_state)
        n = len(l)

        def em_process(B, l, h0):
            h = h0
            m, p = B[0].shape
            s = np.repeat(1, p)

            def NLDD(h, s, B, l):
                res = []
                for i in range(len(B)):
                    res.append(-np.log(1 - np.abs(
                        l[i] - np.max(np.exp(-np.linalg.norm(s * (B[i] - h), axis=1)**2))
                    )))
                res = np.ma.masked_invalid(res).sum()
                return res
            
            nldd0 = np.inf
            nldd1 = NLDD(h, s, B, l)
            while (nldd1 < nldd0):
                ### E step 
                j_max = []
                for i in range(len(B)):
                    j_max.append(np.argmax(np.exp(-np.linalg.norm(s * (B[i] - h), axis=1)**2)))
                ### M step
                def f(hs, *args):
                    B = args[0]
                    l = args[1]
                    j_max = args[2]
                    _, p = B[0].shape
                    res = []
                    for i in range(len(B)):
                        j = j_max[i]
                        res.append(-np.log(1 - np.abs(l[i] - np.exp(-np.linalg.norm(hs[p:] * (B[i][j] - hs[:p]))**2))))
                    res = np.ma.masked_invalid(res).sum()
                    return res
                bounds = [(None, None) if j<p else (0, None) for j in range(2*p)]
                opt_res = minimize(f, np.r_[h,s], args=(B, l, j_max), bounds=bounds)
                if not opt_res.success:
                    print("Optimization failed at M step")
                new_h = opt_res.x[:p].copy()
                new_s = opt_res.x[p:].copy()
                nldd0 = nldd1
                nldd1 = NLDD(new_h, new_s, B, l)
                if nldd1 < nldd0:
                    h, s = new_h, new_s
            return nldd1, h, s
        
        iis = np.random.choice(np.flatnonzero(l == 1), self.n_init_bags)
        best_res = 0
        for i in iis:
            for j in range(len(B[i])):
                _, h, s = em_process(B, l, B[i][j].copy())
                probs = np.empty(n)
                for ii in range(len(B)):
                    probs[ii] = np.max(np.exp(-np.linalg.norm(s * (B[ii] - h), axis=1)**2))
                preds = (probs > 0.5).astype(int)
                score = accuracy_score(l , preds)
                if score > best_res:
                    best_res = score
                    self.h_ = h
                    self.s_ = s
        return self

    def predict(self, B):
        probs = self.predict_proba(B)
        preds = (probs > 0.5).astype(int)
        return preds

    def fit_predict(self, B, l):
        return self.fit(B, l).predict(B)

    def predict_proba(self, B):
        n = len(B)
        probs = np.empty(n)
        for i in range(len(B)):
            probs[i] = np.max(np.exp(-np.linalg.norm(self.s_ * (B[i] - self.h_), axis=1)**2))
        return probs

    def predict_log_proba(self, B):
        return np.log(self.predict_proba(B))
    
    def score(self, B, l):
        l_pred = self.predict(B)
        return accuracy_score(l , l_pred)





class EM_DDv2(BaseEstimator):

    def __init__(self, random_state=None, n_inits=3, tol=1e-3, max_iter=100):
        self.random_state = random_state
        self.n_inits = n_inits
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, B, l):
        np.random.seed(self.random_state)
        m = len(l)

        def em_process(B, l, t0):
            t = t0
            d = B[0].shape[1]
            s = np.repeat(10, d)

            def NLDD(t, s, B, l):
                res = np.empty(len(B))
                B_reprs = None
                for i in range(len(B)):
                    dists = np.linalg.norm(s * (B[i] - t), axis=1)**2
                    j_star = np.argmin(dists)
                    B_ij = np.expand_dims(B[i][j_star], axis=0)
                    B_reprs = B_ij if B_reprs is None else np.r_[B_reprs,B_ij]
                    res[i] = dists[j_star] if l[i]==1 else -np.log(1 - np.exp(-dists[j_star]))
                res = np.ma.masked_invalid(res).sum()
                return res, B_reprs
            
            nldd0 = np.inf
            n_iter = 0
            ### First E step
            nldd1, B_reprs = NLDD(t, s, B, l)
            while (nldd1 < nldd0) and (n_iter < self.max_iter):
                ### M step
                def f(ts, *args):
                    X, l = args[0], args[1]
                    _, d = X.shape
                    dists = np.linalg.norm(ts[d:] * (X - ts[:d]), axis=1)**2
                    res = np.sum(dists[l==1])
                    res -= np.sum(np.ma.masked_invalid(np.log(1 - np.exp(-dists[l==0]))))
                    return res
                def jac_f(ts, *args):
                    X, l = args[0], args[1]
                    _, d = X.shape
                    res = np.zeros(2*d)
                    probs_neg_bags = np.expand_dims(np.exp(-np.linalg.norm(ts[d:] * (X[l==0] - ts[:d]), axis=1)**2), axis=1)
                    res[:d] += 2 * ts[d:]**2 * np.sum(np.ma.masked_invalid((X[l==0] - ts[:d]) * probs_neg_bags / (1 - probs_neg_bags)), axis=0)
                    res[:d] -= 2 * ts[d:]**2 * np.sum((X[l==1] - ts[:d]), axis=0)
                    res[d:] -= 2 * s * np.sum(np.ma.masked_invalid((X[l==0] - t)**2 * probs_neg_bags / (1 - probs_neg_bags)), axis=0)
                    res[d:] += 2 * s * np.sum((X[l==1] - t)**2, axis=0)
                    return res
                bounds = [(None, None) if j<d else (0.1, None) for j in range(2*d)]
                opt_res = minimize(f, np.r_[t,s], args=(B_reprs, l, s), jac=jac_f, bounds=bounds)
                if not opt_res.success:
                    print("Optimization failed at M step")
                new_t = opt_res.x[:d].copy()
                new_s = opt_res.x[d:].copy()
                nldd0 = nldd1
                ### E step
                nldd1, B_reprs = NLDD(new_t, new_s, B, l)
                if nldd1 < nldd0:
                    t, s = new_t, new_s
                n_iter += 1
            return t, s
            

        ### Look for init points
        iis = np.flatnonzero(l == 1)
        X_pos = None
        for i in iis:
            X_pos = B[i].copy() if X_pos is None else np.r_[X_pos, B[i]]
        kmeans = KMeans(n_clusters=self.n_inits, n_init=1).fit(StandardScaler().fit_transform(X_pos))

        best_res = 0
        for k in range(self.n_inits):
            t0 = X_pos[kmeans.labels_==k].mean(axis=0)
            t, s = em_process(B, l, t0)
            probs = np.empty(m)
            for ii in range(len(B)):
                probs[ii] = np.max(np.exp(-np.linalg.norm(s * (B[ii] - t), axis=1)**2))
            preds = (probs > 0.5).astype(int)
            score = accuracy_score(l , preds)
            if score > best_res:
                best_res = score
                self.t_ = t
                self.s_ = s
        return self

    def predict(self, B):
        probs = self.predict_proba(B)
        preds = (probs > 0.5).astype(int)
        return preds

    def fit_predict(self, B, l):
        return self.fit(B, l).predict(B)

    def predict_proba(self, B):
        n = len(B)
        probs = np.empty(n)
        for i in range(len(B)):
            probs[i] = np.max(np.exp(-np.linalg.norm(self.s_ * (B[i] - self.t_), axis=1)**2))
        return probs

    def predict_log_proba(self, B):
        return np.log(self.predict_proba(B))
    
    def score(self, B, l):
        l_pred = self.predict(B)
        return accuracy_score(l , l_pred)