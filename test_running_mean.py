import numpy as np


class RunningMeanStd:
    def __init__(self):
        self.mean = np.zeros(1)
        self.var = np.ones(1)
        self.count = 1e-4


def update_rms(rms, batch):
    batch_mean = np.mean(batch, axis=0)
    batch_var = np.var(batch, axis=0)
    batch_count = batch.shape[0]

    delta = batch_mean - rms.mean
    tot_count = rms.count + batch_count

    new_mean = rms.mean + delta * (batch_count / tot_count)
    m_a = rms.var * rms.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + np.square(delta) * rms.count * (batch_count / tot_count)
    new_var = m2 / tot_count

    rms.mean = new_mean
    rms.var = new_var
    rms.count = tot_count
    return rms


rms = RunningMeanStd()
# Feed [1.0, 1.0, 1.0] in three batches
for _ in range(3):
    update_rms(rms, np.array([1.0]).reshape(1, 1))

print(f"Mean: {rms.mean}, Var: {rms.var}")
