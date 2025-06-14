def test_faiss():
    import faiss
    import numpy as np

    train_x = np.random.uniform(size=(2, 5))
    index = faiss.IndexFlatL2(train_x.shape[-1])
    index.add(train_x)

    x_q = np.random.uniform(size=(1, 5))
    # search returns squared distance
    d2, idx = index.search(x_q, k=2)
    assert idx.min() == 0
    assert idx.max() == 1

    d2 = d2.flatten()[idx.flatten()]

    for i_x, x in enumerate(train_x):
        assert np.abs(((x - x_q) ** 2).sum() - d2[i_x]) < 1e-6
