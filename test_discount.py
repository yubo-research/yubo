def run_baselines(rewards, dones, gamma):
    ret = 0.0
    returns = []
    for r, d in zip(rewards, dones):
        ret = ret * gamma + r
        returns.append(ret)
        if d:
            ret = 0.0
    return returns


def run_mjx(rewards, dones, gamma):
    ret = 0.0
    returns = []
    for r, d in zip(rewards, dones):
        ret = r + gamma * ret * (1.0 - d)
        returns.append(ret)
    return returns


rewards = [1.0, 2.0, 3.0]
dones = [False, True, False]
gamma = 0.99

print("Baselines:", run_baselines(rewards, dones, gamma))
print("MJX/TorchRL:", run_mjx(rewards, dones, gamma))
