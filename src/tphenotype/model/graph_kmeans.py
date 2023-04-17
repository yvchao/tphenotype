import numpy as np

from ..utils.utils import batch_d


def d_js(p, q):
    if len(p.shape) == 1:
        p = p[np.newaxis, :]

    if len(q.shape) == 1:
        q = q[np.newaxis, :]

    d = batch_d(p, q)
    return d


def initialize_centers(S, K):
    # S: N x N distance matrix, symmetrical
    N, _ = S.shape

    cluster_assignment = np.full((N,), -1)

    core_indicies = []
    # cluster 1
    total_distance = np.sum(S, axis=-1)
    idx = np.argmin(total_distance)
    cluster_assignment[idx] = 0
    core_indicies.append(idx)
    sample_idx = np.arange(N)

    for k in range(1, K):
        mask = np.isin(sample_idx, core_indicies)
        candidate_idx = sample_idx[~mask]
        total_distance = np.sum(S[candidate_idx][:, core_indicies], axis=-1)
        idx = np.argmax(total_distance)
        sample_sel = candidate_idx[idx]
        cluster_assignment[sample_sel] = k
        core_indicies.append(sample_sel)

    for i, cluster in enumerate(cluster_assignment):
        if cluster != -1:
            continue
        delta_J = np.zeros((K,))
        for k in range(K):
            (idx,) = np.where(cluster_assignment == k)
            delta_J = 2 * np.sum(S[i, idx])
        cluster_assignment[i] = np.argmin(delta_J)

    return cluster_assignment


class GraphKmeans:
    def __init__(self, K, S, Q):
        N, y_dim = Q.shape
        self.K = K
        self.Q = Q
        self.S = S
        self.delta = np.log(2)

        self.cluster_assignment = np.full((N,), -1)
        self.cluster_centroids = np.full((self.K, y_dim), np.nan)

    def _init_clusters(self):
        delta = 0.0
        for k in range(self.K):
            (idx,) = np.where(self.cluster_assignment == k)
            probs_k = self.Q[idx]
            # update cluster centroid
            self.cluster_centroids[k] = np.mean(probs_k, axis=0)
            dist = d_js(probs_k, self.cluster_centroids[k])[:, 0]
            # update delta
            delta = max(delta, np.max(dist))
            # representative sample that is closest to centroid
            idx_sel = np.argmin(dist)
            # only keep one sample in cluster
            self.cluster_assignment[idx] = -1
            self.cluster_assignment[idx[idx_sel]] = k

        # update delta
        self.delta = min(self.delta, 2 * delta)

        (self.idx_assigned,) = np.where(self.cluster_assignment != -1)  # pylint: disable=attribute-defined-outside-init
        (self.idx_free,) = np.where(self.cluster_assignment == -1)  # pylint: disable=attribute-defined-outside-init

    def fit(self, max_iter=1000, tol=1e-7, patience=5):
        best_loss = np.inf
        no_improvement_count = 0
        best_clusters = self.cluster_assignment.copy()
        best_centroids = self.cluster_centroids.copy()

        # initialize clusters via approximate solution to upper bound minimization
        self.cluster_assignment = initialize_centers(self.S, self.K)

        for i in range(max_iter):  # pylint: disable=unused-variable
            self._init_clusters()
            # create similarity graph G
            G = 1.0 * (self.S <= self.delta)
            while True:
                idx_candidates = self._get_candidates(G)
                if len(idx_candidates) == 0:
                    break

                reachable_clusters = self._get_reachable_clusters(G, idx_candidates)
                self._assign_cluster(idx_candidates, reachable_clusters)

            self._update_centroids()

            loss = self._calculate_loss()
            if abs(best_loss - loss) < tol:
                break

            if loss < best_loss:
                best_loss = loss
                best_clusters = self.cluster_assignment.copy()
                best_centroids = self.cluster_centroids.copy()
            else:
                no_improvement_count += 1

            if no_improvement_count > patience:
                break

        self.cluster_assignment = best_clusters
        self.cluster_centroids = best_centroids
        (self.idx_assigned,) = np.where(self.cluster_assignment != -1)  # pylint: disable=attribute-defined-outside-init
        (self.idx_free,) = np.where(self.cluster_assignment == -1)  # pylint: disable=attribute-defined-outside-init

        return self

    def _get_candidates(self, G):
        reachable = np.any(G[self.idx_assigned][:, self.idx_free], axis=0)
        (idx_candidates,) = np.where(reachable)
        return self.idx_free[idx_candidates]

    def _get_reachable_clusters(self, G, candidates):
        reachable = [np.where(G[self.idx_assigned][:, node])[0] for node in candidates]
        reachable_clusters = [np.unique(self.cluster_assignment[self.idx_assigned][node]) for node in reachable]
        return reachable_clusters

    def _assign_cluster(self, candidates, reachable_clusters):
        for node, reachable in zip(candidates, reachable_clusters):
            p = self.Q[node]
            p_clusters = self.cluster_centroids[reachable]
            dist = d_js(p, p_clusters)[0]
            self.cluster_assignment[node] = reachable[np.argmin(dist)]
        (self.idx_assigned,) = np.where(self.cluster_assignment != -1)  # pylint: disable=attribute-defined-outside-init
        (self.idx_free,) = np.where(self.cluster_assignment == -1)  # pylint: disable=attribute-defined-outside-init

    def _update_centroids(self):
        for i in range(self.K):
            (cluster_i,) = np.where(self.cluster_assignment == i)
            self.cluster_centroids[i] = np.mean(self.Q[cluster_i], axis=0)

    def _calculate_loss(self):
        loss = 0
        for i in range(self.K):
            (cluster_i,) = np.where(self.cluster_assignment == i)
            P = self.Q[cluster_i]
            P_c = self.cluster_centroids[i]
            dist = d_js(P, P_c)[:, 0]
            loss += np.sum(dist)
        return loss

    def get_cluster(self, k):
        (cluster_k,) = np.where(self.cluster_assignment == k)
        p = self.cluster_centroids[k]
        return {"samples": cluster_k, "p": p}

    def get_outliers(self):
        return self.idx_free

    def predict(self, S, probs):
        affinity = 1.0 * (S <= self.delta)
        cluster_assignment = np.full((len(probs),), -1)
        for i, p in enumerate(probs):
            reachable_clusters = self.cluster_assignment[affinity[i] == 1]
            reachable_clusters = np.unique(reachable_clusters)
            if len(reachable_clusters) > 1:
                p_clusters = self.cluster_centroids[reachable_clusters]
                dist = d_js(p, p_clusters)[0]
                cluster_assignment[i] = reachable_clusters[np.argmin(dist)]
            elif len(reachable_clusters) == 1:
                cluster_assignment[i] = reachable_clusters.item()
            else:
                pass

        return cluster_assignment
