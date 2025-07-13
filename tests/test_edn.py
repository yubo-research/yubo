import numpy as np
import pytest

from model.edn import EpistemicNovelty


def test_edn_initialization():
    k_novelty = 5
    edn = EpistemicNovelty(k_novelty)

    assert edn.k_novelty == k_novelty
    assert edn.descriptors is None
    assert edn.descriptor_ses is None
    assert edn.index is None


def test_add_first_descriptor():
    edn = EpistemicNovelty(k_novelty=3)

    desc = np.array([[1.0, 2.0, 3.0]])
    desc_se = np.array([[0.1, 0.2, 0.3]])

    edn.add(desc, desc_se)

    assert edn.descriptors is not None
    assert edn.descriptor_ses is not None
    assert edn.descriptors.shape == (1, 3)
    assert edn.descriptor_ses.shape == (1, 3)
    assert np.array_equal(edn.descriptors[0], desc[0])
    assert np.array_equal(edn.descriptor_ses[0], desc_se[0])
    assert edn.index is not None
    assert edn.index.ntotal == 1


def test_add_multiple_descriptors():
    edn = EpistemicNovelty(k_novelty=3)

    desc1 = np.array([[1.0, 2.0, 3.0]])
    desc_se1 = np.array([[0.1, 0.2, 0.3]])

    desc2 = np.array([[4.0, 5.0, 6.0]])
    desc_se2 = np.array([[0.4, 0.5, 0.6]])

    edn.add(desc1, desc_se1)
    edn.add(desc2, desc_se2)

    assert edn.descriptors.shape == (2, 3)
    assert edn.descriptor_ses.shape == (2, 3)
    assert np.array_equal(edn.descriptors[0], desc1[0])
    assert np.array_equal(edn.descriptors[1], desc2[0])
    assert np.array_equal(edn.descriptor_ses[0], desc_se1[0])
    assert np.array_equal(edn.descriptor_ses[1], desc_se2[0])
    assert edn.index.ntotal == 2


def test_add_multiple_rows():
    edn = EpistemicNovelty(k_novelty=3)

    descs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    desc_ses = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    edn.add(descs, desc_ses)

    assert edn.descriptors.shape == (2, 3)
    assert edn.descriptor_ses.shape == (2, 3)
    assert np.array_equal(edn.descriptors, descs)
    assert np.array_equal(edn.descriptor_ses, desc_ses)
    assert edn.index.ntotal == 2


def test_add_different_dimensions():
    edn = EpistemicNovelty(k_novelty=3)

    desc1 = np.array([[1.0, 2.0]])
    desc_se1 = np.array([[0.1, 0.2]])

    desc2 = np.array([[3.0, 4.0, 5.0]])
    desc_se2 = np.array([[0.3, 0.4, 0.5]])

    edn.add(desc1, desc_se1)

    with pytest.raises(AssertionError):
        edn.add(desc2, desc_se2)


def test_add_1d_array_error():
    edn = EpistemicNovelty(k_novelty=3)

    desc = np.array([1.0, 2.0, 3.0])
    desc_se = np.array([0.1, 0.2, 0.3])

    with pytest.raises(AssertionError):
        edn.add(desc, desc_se)


def test_faiss_index_creation():
    edn = EpistemicNovelty(k_novelty=3)

    desc = np.array([[1.0, 2.0, 3.0, 4.0]])
    desc_se = np.array([[0.1, 0.2, 0.3, 0.4]])

    edn.add(desc, desc_se)

    assert edn.index is not None
    assert edn.index.d == 4
    assert edn.index.ntotal == 1
    assert isinstance(edn.index, type(edn.index))


def test_empty_state():
    edn = EpistemicNovelty(k_novelty=3)

    assert edn.descriptors is None
    assert edn.descriptor_ses is None
    assert edn.index is None


def test_posterior_empty():
    edn = EpistemicNovelty(k_novelty=3)

    mean, se = edn.dominated_novelty_of_last_addition()
    assert mean is None
    assert se is None


def test_posterior_single_descriptor():
    edn = EpistemicNovelty(k_novelty=3)

    desc = np.array([[1.0, 2.0, 3.0]])
    desc_se = np.array([[0.1, 0.2, 0.3]])
    edn.add(desc, desc_se)

    mean, se = edn.dominated_novelty_of_last_addition()
    assert mean is None
    assert se is None


def test_posterior_multiple_descriptors():
    edn = EpistemicNovelty(k_novelty=3)

    desc1 = np.array([[0.0, 0.0, 0.0]])
    desc_se1 = np.array([[0.1, 0.1, 0.1]])

    desc2 = np.array([[1.0, 0.0, 0.0]])
    desc_se2 = np.array([[0.1, 0.1, 0.1]])

    desc3 = np.array([[2.0, 0.0, 0.0]])
    desc_se3 = np.array([[0.1, 0.1, 0.1]])

    edn.add(desc1, desc_se1)
    edn.add(desc2, desc_se2)
    edn.add(desc3, desc_se3)

    mean, se = edn.dominated_novelty_of_last_addition(k_novelty=2)
    assert mean is not None
    assert se is not None
    assert mean > 0
    assert se >= 0


def test_posterior_precision_weighting():
    edn = EpistemicNovelty(k_novelty=3)

    desc1 = np.array([[0.0, 0.0]])
    desc_se1 = np.array([[0.1, 0.1]])

    desc2 = np.array([[1.0, 0.0]])
    desc_se2 = np.array([[0.1, 0.1]])

    desc3 = np.array([[2.0, 0.0]])
    desc_se3 = np.array([[1.0, 1.0]])

    edn.add(desc1, desc_se1)
    edn.add(desc2, desc_se2)
    edn.add(desc3, desc_se3)

    mean, se = edn.dominated_novelty_of_last_addition(k_novelty=3)
    assert mean is not None
    assert se is not None

    expected_distance_to_desc2 = 1.0

    assert abs(mean - expected_distance_to_desc2) < 0.5


def test_posterior_index_out_of_range():
    edn = EpistemicNovelty(k_novelty=3)

    desc1 = np.array([[1.0, 2.0, 3.0]])
    desc_se1 = np.array([[0.1, 0.2, 0.3]])
    desc2 = np.array([[4.0, 5.0, 6.0]])
    desc_se2 = np.array([[0.4, 0.5, 0.6]])

    edn.add(desc1, desc_se1)
    edn.add(desc2, desc_se2)

    mean, se = edn.dominated_novelty_of_last_addition()
    assert mean is not None
    assert se is not None


def test_novelty_single_sample():
    edn = EpistemicNovelty(k_novelty=3)

    desc1 = np.array([[0.0, 0.0, 0.0]])
    desc_se1 = np.array([[0.1, 0.1, 0.1]])

    desc2 = np.array([[1.0, 0.0, 0.0]])
    desc_se2 = np.array([[0.1, 0.1, 0.1]])

    desc3 = np.array([[2.0, 0.0, 0.0]])
    desc_se3 = np.array([[0.1, 0.1, 0.1]])

    edn.add(desc1, desc_se1)
    edn.add(desc2, desc_se2)
    edn.add(desc3, desc_se3)

    test_desc = np.array([[0.5, 0.0, 0.0]])
    test_desc_se = np.array([[0.1, 0.1, 0.1]])

    novelty, novelty_se = edn.novelty(test_desc, test_desc_se, k_novelty=2)

    assert novelty is not None
    assert novelty_se is not None
    assert novelty > 0
    assert novelty_se >= 0
    assert novelty.shape == ()
    assert novelty_se.shape == ()


def test_novelty_multiple_samples():
    edn = EpistemicNovelty(k_novelty=3)

    desc1 = np.array([[0.0, 0.0, 0.0]])
    desc_se1 = np.array([[0.1, 0.1, 0.1]])

    desc2 = np.array([[1.0, 0.0, 0.0]])
    desc_se2 = np.array([[0.1, 0.1, 0.1]])

    desc3 = np.array([[2.0, 0.0, 0.0]])
    desc_se3 = np.array([[0.1, 0.1, 0.1]])

    edn.add(desc1, desc_se1)
    edn.add(desc2, desc_se2)
    edn.add(desc3, desc_se3)

    test_descs = np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.0, 0.0]])
    test_desc_ses = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])

    novelty, novelty_se = edn.novelty(test_descs, test_desc_ses, k_novelty=2)

    assert novelty is not None
    assert novelty_se is not None
    assert novelty.shape == (3,)
    assert novelty_se.shape == (3,)
    assert np.all(novelty > 0)
    assert np.all(novelty_se >= 0)


def test_novelty_empty_state():
    edn = EpistemicNovelty(k_novelty=3)

    test_desc = np.array([[0.5, 0.0, 0.0]])
    test_desc_se = np.array([[0.1, 0.1, 0.1]])

    novelty, novelty_se = edn.novelty(test_desc, test_desc_se)

    assert novelty is None
    assert novelty_se is None


def test_novelty_single_descriptor():
    edn = EpistemicNovelty(k_novelty=3)

    desc1 = np.array([[0.0, 0.0, 0.0]])
    desc_se1 = np.array([[0.1, 0.1, 0.1]])
    edn.add(desc1, desc_se1)

    test_desc = np.array([[0.5, 0.0, 0.0]])
    test_desc_se = np.array([[0.1, 0.1, 0.1]])

    novelty, novelty_se = edn.novelty(test_desc, test_desc_se)

    assert novelty is None
    assert novelty_se is None


def test_novelty_exclude_nearest():
    edn = EpistemicNovelty(k_novelty=3)

    desc1 = np.array([[0.0, 0.0, 0.0]])
    desc_se1 = np.array([[0.1, 0.1, 0.1]])

    desc2 = np.array([[1.0, 0.0, 0.0]])
    desc_se2 = np.array([[0.1, 0.1, 0.1]])

    desc3 = np.array([[2.0, 0.0, 0.0]])
    desc_se3 = np.array([[0.1, 0.1, 0.1]])

    edn.add(desc1, desc_se1)
    edn.add(desc2, desc_se2)
    edn.add(desc3, desc_se3)

    test_desc = np.array([[0.5, 0.0, 0.0]])
    test_desc_se = np.array([[0.1, 0.1, 0.1]])

    novelty_with_exclude, novelty_se_with_exclude = edn.novelty(test_desc, test_desc_se, k_novelty=2, exclude_nearest=True)
    novelty_without_exclude, novelty_se_without_exclude = edn.novelty(test_desc, test_desc_se, k_novelty=2, exclude_nearest=False)

    assert novelty_with_exclude is not None
    assert novelty_without_exclude is not None
    assert novelty_with_exclude != novelty_without_exclude


def test_novelty_custom_k_novelty():
    edn = EpistemicNovelty(k_novelty=5)

    desc1 = np.array([[0.0, 0.0, 0.0]])
    desc_se1 = np.array([[0.1, 0.1, 0.1]])

    desc2 = np.array([[1.0, 0.0, 0.0]])
    desc_se2 = np.array([[0.1, 0.1, 0.1]])

    desc3 = np.array([[2.0, 0.0, 0.0]])
    desc_se3 = np.array([[0.1, 0.1, 0.1]])

    edn.add(desc1, desc_se1)
    edn.add(desc2, desc_se2)
    edn.add(desc3, desc_se3)

    test_desc = np.array([[0.5, 0.0, 0.0]])
    test_desc_se = np.array([[0.1, 0.1, 0.1]])

    novelty_default, novelty_se_default = edn.novelty(test_desc, test_desc_se)
    novelty_custom, novelty_se_custom = edn.novelty(test_desc, test_desc_se, k_novelty=2)

    assert novelty_default is not None
    assert novelty_custom is not None
    assert novelty_default != novelty_custom
