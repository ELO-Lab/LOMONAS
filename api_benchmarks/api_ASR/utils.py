import hashlib
import numpy as np
import copy

all_ops = ['linear', 'conv5', 'conv5d2', 'conv7', 'conv7d2', 'zero']

def arch_int_to_vec(arch_int):
    arch_vec = [(arch_int[0], arch_int[1]), (arch_int[2], arch_int[3], arch_int[4]),
                (arch_int[5], arch_int[6], arch_int[7], arch_int[8])]
    return arch_vec

def get_model_graph(arch_vec, ops=None, minimize=True, keep_dims=False):
    if ops is None:
        ops = all_ops
    num_nodes = len(arch_vec)
    mat = np.zeros((num_nodes+2, num_nodes+2))
    labels = ['input']
    prev_skips = []
    for nidx, node in enumerate(arch_vec):
        op = node[0]
        labels.append(ops[op])
        mat[nidx, nidx+1] = 1
        for i, sc in enumerate(prev_skips):
            if sc:
                mat[i, nidx+1] = 1
        prev_skips = node[1:]
    labels.append('output')
    mat[num_nodes, num_nodes+1] = 1
    for i, sc in enumerate(prev_skips):
        if sc:
            mat[i, num_nodes+1] = 1
    orig = None
    if minimize:
        orig = copy.copy(mat), copy.copy(labels)
        for n in range(len(mat)):
            if labels[n] == 'zero':
                for n2 in range(len(mat)):
                    if mat[n, n2]:
                        mat[n, n2] = 0
                    if mat[n2, n]:
                        mat[n2, n] = 0
        def bfs(src, mat, backward):
            visited = np.zeros(len(mat))
            q = [src]
            visited[src] = 1
            while q:
                n = q.pop()
                for n2 in range(len(mat)):
                    if visited[n2]:
                        continue
                    if (backward and mat[n2, n]) or (not backward and mat[n, n2]):
                        q.append(n2)
                        visited[n2] = 1
            return visited
        vfw = bfs(0, mat, False)
        vbw = bfs(len(mat)-1, mat, True)
        v = vfw + vbw
        dangling = (v < 2).nonzero()[0]
        if dangling.size:
            if keep_dims:
                mat[dangling, :] = 0
                mat[:, dangling] = 0
                for i in dangling:
                    labels[i] = None
            else:
                mat = np.delete(mat, dangling, axis=0)
                mat = np.delete(mat, dangling, axis=1)
                for i in sorted(dangling, reverse=True):
                    del labels[i]
    return (mat, labels), orig

def graph_hash(g):
    m, l = g
    def hash_module(matrix, labelling):
        """Computes a graph-invariance MD5 hash of the matrix and label pair.
        Args:
            matrix: np.ndarray square upper-triangular adjacency matrix.
            labelling: list of int labels of length equal to both dimensions of
                matrix.
        Returns:
            MD5 hash of the matrix and labelling.
        """
        vertices = np.shape(matrix)[0]
        in_edges = np.sum(matrix, axis=0).tolist()
        out_edges = np.sum(matrix, axis=1).tolist()
        assert len(in_edges) == len(out_edges) == len(labelling), f'{labelling} {matrix}'
        hashes = list(zip(out_edges, in_edges, labelling))
        hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
        # Computing this up to the diameter is probably sufficient but since the
        # operation is fast, it is okay to repeat more times.
        for _ in range(vertices):
            new_hashes = []
            for v in range(vertices):
                in_neighbours = [hashes[w] for w in range(vertices) if matrix[w, v]]
                out_neighbours = [hashes[w] for w in range(vertices) if matrix[v, w]]
                new_hashes.append(hashlib.md5(
                        (''.join(sorted(in_neighbours)) + '|' +
                        ''.join(sorted(out_neighbours)) + '|' +
                        hashes[v]).encode('utf-8')).hexdigest())
            hashes = new_hashes
        fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()
        return fingerprint
    labels = []
    if l:
        labels = [-1] + [all_ops.index(op) for op in l[1:-1]] + [-2]
    return hash_module(m, labels)

def get_model_hash(arch_vec, ops=None, minimize=True):
    """
    Get hash of the architecture specified by arch_vec.
    Architecture hash can be used to determine if two configurations from the search space are in fact the same (graph isomorphism).
    """
    g, _ = get_model_graph(arch_vec, ops=ops, minimize=minimize)
    return graph_hash(g)

def get_hashKey(arch_int):
    arch_vec = arch_int_to_vec(arch_int)
    hashKey = get_model_hash(arch_vec)
    return hashKey