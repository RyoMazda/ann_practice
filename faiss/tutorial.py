# https://github.com/facebookresearch/faiss/wiki/Getting-started
import faiss
import numpy as np


if __name__ == '__main__':
    np.random.seed(1234)             # make reproducible
    d = 64                           # dimension

    nb = 100_000                      # database size
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.

    # Building an index and adding the vectors to it
    index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)
    index.add(xb)                  # add vectors to the index
    print(index.ntotal)

    # Searching
    nq = 1_000                       # nb of queries
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.
    k = 4                          # we want to see 4 nearest neighbors
    D, I = index.search(xb[:5], k) # sanity check
    print(I)
    print(D)
    D, I = index.search(xq, k)     # actual search
    print(I[:5])                   # neighbors of the 5 first queries
    print(I[-5:])                  # neighbors of the 5 last queries
