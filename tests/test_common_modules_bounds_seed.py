import numpy as np


class TestAllBounds:
    def test_constants(self):
        import common.all_bounds as all_bounds

        assert all_bounds.x_low == -1.0
        assert all_bounds.x_high == 1.0
        assert all_bounds.x_width == 2.0

        assert all_bounds.p_low == -1.0
        assert all_bounds.p_high == 1.0
        assert all_bounds.p_width == 2.0

        assert all_bounds.bt_low == 0.0
        assert all_bounds.bt_high == 1.0
        assert all_bounds.bt_width == 1.0

    def test_get_box_bounds_x(self):
        import common.all_bounds as all_bounds

        box = all_bounds.get_box_bounds_x(3)
        assert box.shape == (3,)
        np.testing.assert_array_equal(box.low, [-1.0, -1.0, -1.0])
        np.testing.assert_array_equal(box.high, [1.0, 1.0, 1.0])

    def test_get_box_bounds_x_1d(self):
        import common.all_bounds as all_bounds

        box = all_bounds.get_box_bounds_x(1)
        assert box.shape == (1,)
        np.testing.assert_array_equal(box.low, [-1.0])
        np.testing.assert_array_equal(box.high, [1.0])

    def test_get_box_1d01(self):
        import common.all_bounds as all_bounds

        box = all_bounds.get_box_1d01()
        assert box.low == 0.0
        assert box.high == 1.0


class TestSeedAll:
    def test_seed_all_deterministic(self):
        from common.seed_all import seed_all

        seed_all(42)
        vals1 = [np.random.rand() for _ in range(5)]

        seed_all(42)
        vals2 = [np.random.rand() for _ in range(5)]

        np.testing.assert_array_almost_equal(vals1, vals2)

    def test_seed_all_different_seeds(self):
        from common.seed_all import seed_all

        seed_all(42)
        vals1 = np.random.rand()

        seed_all(123)
        vals2 = np.random.rand()

        assert vals1 != vals2

    def test_seed_all_torch_deterministic(self):
        import torch

        from common.seed_all import seed_all

        seed_all(42)
        t1 = torch.rand(3)

        seed_all(42)
        t2 = torch.rand(3)

        torch.testing.assert_close(t1, t2)
