def test_nds():
    import numpy as np
    from nds import ndomsort

    # 5 objectives, 30 points
    seq = np.random.randint(low=-10, high=10, size=(30, 5))

    print()
    print(seq)
    # fronts[i_front] = [point_0, point_1, ...]
    fronts = ndomsort.non_domin_sort(seq)
    print(type(fronts), fronts)

    # Or we can get values of objectives.
    fronts = ndomsort.non_domin_sort(seq, lambda x: x[:4])
    print(type(fronts), fronts)

    # 'fronts' is a tuple of front's indices, not a dictionary.
    fronts = ndomsort.non_domin_sort(seq, only_front_indices=True)
    print(type(fronts), fronts)

    seq = np.random.randint(low=-10, high=10, size=(300, 5))
    assert (
        ndomsort.non_domin_sort(seq, only_front_indices=True)
        == ndomsort.non_domin_sort(seq, only_front_indices=True)
        == ndomsort.non_domin_sort(seq, only_front_indices=True)
        == ndomsort.non_domin_sort(seq, only_front_indices=True)
    )
