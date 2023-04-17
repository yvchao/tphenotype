# pylint: disable=attribute-defined-outside-init

import numpy as np
from tqdm import auto

from ..utils.utils import batch_d, select_by_steps
from .graph_kmeans import GraphKmeans


class SyncClustering:
    def __init__(self, predictor, K, steps=(-1,), tol=1e-6, verbose=True, test_num=50, threshold=np.log(2)):
        self.K = K
        self.predictor = predictor
        self.steps = np.array(steps)
        self.tol = tol
        self.test_num = test_num
        self.verbose = verbose
        self.threshold = threshold

        all_non_neg = np.all(self.steps >= 0)
        all_neg = np.all(self.steps < 0)
        assert all_non_neg or all_neg

    def _sub_steps(self, series, mask):
        return select_by_steps(series, mask, self.steps)

    def fit(self, x, t, mask):
        probs_corpus = self.predictor.predict_proba_g(x, t)
        self.probs_corpus = self._sub_steps(probs_corpus, mask)
        x_corpus = self.predictor.encode(x, t)
        self.x_corpus = self._sub_steps(x_corpus, mask)
        # self.x_corpus_ = self.predictor.transform(x, t)
        z_corpus = self.predictor.embed(x, t)
        self.z_corpus = self._sub_steps(z_corpus, mask)

        # if self.verbose:
        #     print('compress corpus set via self-expression')
        # compress the corpus set
        # idx = self._compress_corpus(self.x_corpus, compress_level=1000.0 / len(self.x_corpus))
        # self.x_corpus = self.x_corpus[idx]
        # self.probs_corpus = self.probs_corpus[idx]
        # self.z_corpus = self.z_corpus[idx]
        # self.idx_corpus = idx

        xs = (self.x_corpus, self.x_corpus)
        probs = (self.probs_corpus, self.probs_corpus)
        if self.verbose:
            print("construct similarity graph")
        # prior on similarity
        A = self._similarity_piror_fast(xs, probs, identical=True)
        if self.verbose:
            print("discover clusters from similarity graph")
        self.A_corpus = A
        self.kmeans = GraphKmeans(self.K, self.A_corpus, self.probs_corpus)
        self.kmeans.fit()
        self.clusters = [self.kmeans.get_cluster(k) for k in range(self.kmeans.K)]

        self.n_clusters = len(self.clusters)
        labels = np.zeros((len(self.x_corpus),))
        for i, cluster in enumerate(self.clusters):
            labels[cluster["samples"]] = i
        self.label_corpus = labels

    # def _compute_proba(self, probs_test, x_test, z_test, batch_size=1000):
    #     xs = (x_test, self.x_corpus)
    #     probs = (probs_test, self.probs_corpus)
    #     A = self._similarity_piror_fast(xs, probs, identical=False)
    #
    #     n, _ = x_test.shape
    #     indices = np.arange(n, dtype=int)
    #
    #     affinity = []
    #     for k in auto.trange(0, n, batch_size, disable=not self.verbose):
    #         i = indices[k:k + batch_size]
    #
    #         Z = (z_test[i], self.z_corpus)
    #         aff = self._get_affinity(Z, A[i], identical=False)
    #         affinity.append(aff)
    #     affinity = np.concatenate(affinity, axis=0)
    #
    #     proba = np.zeros((len(x_test), self.n_clusters))
    #     for i in range(self.n_clusters):
    #         mask = self.label_corpus == i
    #         proba[:, i] = np.sum(affinity[:, mask], axis=1)
    #     return proba

    def _compute_proba_kmeans(self, probs_test, x_test, z_test, batch_size=1000):  # pylint: disable=unused-argument
        xs = (x_test, self.x_corpus)
        probs = (probs_test, self.probs_corpus)
        A = self._similarity_piror_fast(xs, probs, identical=False)

        n, _ = x_test.shape
        indices = np.arange(n, dtype=int)

        clusters = []
        for k in auto.trange(0, n, batch_size, disable=not self.verbose):
            i = indices[k : k + batch_size]
            cluster = self.kmeans.predict(A[i], probs_test[i])
            clusters.append(cluster)
        clusters = np.concatenate(clusters, axis=0)

        proba = np.zeros((len(x_test), self.n_clusters))
        # outliers without cluster assignment
        proba[clusters == -1] = 1.0 / self.n_clusters
        for i in range(self.n_clusters):
            mask = clusters == i
            proba[mask, i] = 1.0
        return proba

    def predict_proba(self, x, t, mask=None, steps=None):
        verbose = self.verbose
        if verbose:
            self.verbose = False

        probs_test = self.predictor.predict_proba_g(x, t)
        x_test = self.predictor.encode(x, t)
        z_test = self.predictor.embed(x, t)

        if mask is not None and steps is not None:
            probs_test = select_by_steps(probs_test, mask, steps)[:, np.newaxis, :]
            x_test = select_by_steps(x_test, mask, steps)[:, np.newaxis, :]
            z_test = select_by_steps(z_test, mask, steps)[:, np.newaxis, :]
        elif steps is not None:
            probs_test = probs_test[:, steps, :]
            x_test = x_test[:, steps, :]
            z_test = z_test[:, steps, :]
        else:
            pass

        _, series_size, _ = x_test.shape
        probs = []
        for s in range(series_size):
            # proba = self._compute_proba(probs_test[:, s, :], x_test[:, s, :], z_test[:, s, :])
            proba = self._compute_proba_kmeans(probs_test[:, s, :], x_test[:, s, :], z_test[:, s, :])
            probs.append(proba)
        probs = np.stack(probs, axis=1)

        if series_size == 1:
            probs = probs[:, 0, :]

        self.verbose = verbose
        return probs

    def predict(self, x, t, mask=None, steps=None):
        proba = self.predict_proba(x, t, mask, steps)
        label = np.argmax(proba, axis=-1)
        return label

    # def _path_gen(self, x1, x2):
    #     t = np.linspace(0, 1, num=self.test_num).reshape((1, 1, -1, 1)).astype(x1.dtype)
    #     path = (1 - t) * x1[..., :, np.newaxis, np.newaxis, :] + t * x2[..., np.newaxis, :, np.newaxis, :]
    #     return path

    def _batch_path_gen(self, x1, x2):
        # x1: batch_size x x_dim
        # x2: batch_size x x_dim
        t = np.linspace(0, 1, num=self.test_num).reshape((1, -1, 1)).astype(x1.dtype)
        path = (1 - t) * x1[:, np.newaxis, :] + t * x2[:, np.newaxis, :]
        return path

    def _batch_path_test(self, x1, x2):
        # x1: batch_size x x_dim
        # x2: batch_size x x_dim

        # paths: batch_size x self.test_num x x_dim
        paths = self._batch_path_gen(x1, x2)
        probs = self.predictor.predict_proba_from_x_rep(paths)
        # d_paths: batch_size x self.test_num
        # d_paths1 = batch_KL(probs[:], probs[:, [0]])[:, :, 0]
        # d_paths2 = batch_KL(probs[:, [0]], probs[:, :])[:, 0, :]
        # d_paths = 0.5 * (d_paths1 + d_paths2)
        # batch_size x test_num
        d_paths1 = batch_d(probs, probs[:, [0]])[:, :, 0]
        d_paths2 = batch_d(probs, probs[:, [-1]])[:, :, 0]
        # batch_size x 2*test_num
        d_paths = np.concatenate([d_paths1, d_paths2], axis=-1)
        # equivalent: batch_size
        dist = np.max(d_paths, axis=-1)
        return dist

    def _select_relevant_samples(self, probs_test, probs_corpus, identical=False):
        dist = batch_d(probs_test, probs_corpus)
        if identical:
            np.fill_diagonal(dist, 0.0)
        # if not identical:
        #     dist = 0.5 * (batch_KL(probs_test, probs_corpus) + batch_KL(probs_corpus, probs_test).T)
        # else:
        #     dist = batch_KL(probs_test, probs_corpus)
        #     dist = (dist + dist.T) / 2
        #     np.fill_diagonal(dist, np.nan)

        # set similarity of x_i, x_j to 1 if prob_i ~ prob_j
        mask = dist <= self.threshold

        # identify connected elements
        I, J = np.where(mask)
        if identical:
            # reduce computation
            m = I < J
            I = I[m]  # noqa: E741
            J = J[m]

        return I, J

    def _similarity_piror_fast(self, xs, probs, identical=False, batch_size=600):
        # x: batch_size, x_dim
        # probs: batch_size, y_dim
        x_test, x_corpus = xs
        probs_test, probs_corpus = probs

        I, J = self._select_relevant_samples(probs_test, probs_corpus, identical)

        test_size, _ = x_test.shape
        corpus_size, _ = x_corpus.shape
        A = np.zeros((test_size, corpus_size), dtype=float)

        indices = np.arange(len(I), dtype=int)

        for k in auto.trange(0, len(indices), batch_size, disable=not self.verbose):
            idx = indices[k : k + batch_size]
            i, j = I[idx], J[idx]
            # batch_dist: test_size x corpus_size
            batch_dist = self._batch_path_test(x_test[i], x_corpus[j])
            A[i, j] = batch_dist
            if identical:
                A[j, i] = A[i, j]

        return A

    # def _connectivity_check(self, x_test, x_corpus, A, kth=7):
    #     # x1 = self.predictor.rep_to_curve(x_test)[:, 0, :]
    #     # x2 = self.predictor.rep_to_curve(x_corpus)[:, 0, :]
    #     # dist = euclidean_distances(x1, x2)
    #     dist = euclidean_distances(x_test, x_corpus)
    #     distant_idx = np.argpartition(dist, kth=kth, axis=-1)[:, kth:]
    #     np.put_along_axis(A, distant_idx, 0, axis=-1)
    #     return A
    #
    # def _batch_compress_corpus(self, batch_corpus, compress_level=0.4):
    #     # batch_corpus: batch_size x corpus_dim
    #     n, _ = batch_corpus.shape
    #     output_size = int(n * compress_level)
    #     A = np.ones((n, n), dtype=batch_corpus.dtype)
    #     np.fill_diagonal(A, 0)
    #     solver = AffinitySolver(n, n)
    #     solver.solve((batch_corpus, batch_corpus), A, verbose=False)
    #     B = solver.get_affinity(A)
    #     contribs = np.sum(B, axis=0)
    #     idx = np.argsort(contribs)[-output_size:]
    #     return idx
    #
    # def _compress_corpus(self, x_corpus, compress_level=0.4, batch_size=300):
    #     # x_corpus: sample_size x x_dim
    #     n, _ = x_corpus.shape
    #     indices = np.arange(n, dtype=int)
    #
    #     sel_idx = []
    #     for k in auto.trange(0, n, batch_size, disable=not self.verbose):
    #         i = indices[k:k + batch_size]
    #         idx = self._batch_compress_corpus(x_corpus[i], compress_level)
    #         sel_idx.append(idx + k)
    #     sel_idx = np.concatenate(sel_idx, axis=0)
    #     return sel_idx

    # def _get_affinity(self, Z, A, identical=False):
    #     if identical:
    #         np.fill_diagonal(A, 0)
    #     z_test, z_corpus = Z
    #     m, _ = z_test.shape
    #     n, _ = z_corpus.shape
    #     solver = AffinitySolver(m, n)
    #     solver.solve((z_test, z_corpus), A, tol=self.tol, verbose=self.verbose)
    #     B = solver.get_affinity(A)
    #     return B
