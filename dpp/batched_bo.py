import numpy as np
import torch


class Batched_BO:
    # Util class for Batched BO algorithms, regardless of being UCB or TS

    def isin_hal(self, xnext, x_hal):
        for v in x_hal:
            if torch.norm(v - xnext) < self.epsilon:
                return True

    def lower_confidence(self, xtest, beta=1.0):  # LCB
        (ymean, yvar) = self.GP.mean_var(xtest)
        np.nan_to_num(yvar)
        conf = ymean - np.sqrt(beta) * yvar
        return conf

    def best_inference(self, xtest, beta=1.0):
        """
        Best conservative guess for maximizer, used to compute inference regret
        """
        if xtest is None:
            # raise ValueError("To get inference regret, must use finite grid method")
            xtest = self.fake_xtest
        conf = self.lower_confidence(xtest, beta)
        i = np.argmax(conf)
        xnext = xtest[i]
        xnext = xnext.view(-1, 1)
        return (xnext, conf[i])

    def eff_d(self, K, s):
        return torch.trace(K @ torch.inverse(K + s * torch.eye(K.shape[0], dtype=torch.double)))

    def step_update(self, xtest, x_batch, safe, get_inference, log_lik_hist=None):
        (_, first_point_var) = self.GP.mean_var(x_batch[0].view(-1, self.GP.d))

        batch_rewards = torch.tensor([], dtype=torch.double)
        # sample reward for the batch
        for xnext in x_batch:
            xnext = xnext[:, None]
            reward = self.F(torch.t(xnext))
            true_reward = self.true_F(torch.t(xnext))
            if not safe or not self.isin(xnext[:, 0]):
                self.x = torch.cat((self.x, torch.t(xnext)), dim=0)
                self.y = torch.cat((self.y, reward), dim=0)
            if batch_rewards.size()[0] == 0:
                batch_rewards = true_reward
            else:
                batch_rewards = torch.cat((batch_rewards, true_reward), dim=0)

        res = {"batch_rewards": batch_rewards, "x_batch": x_batch, "first_p_var": first_point_var[0][0]}

        if log_lik_hist is not None:
            res["log_lik_hist"] = log_lik_hist

        # Get inference reward
        if get_inference:
            (xinfer, _) = self.best_inference(xtest)
            infer_true_reward = self.true_F(torch.t(xinfer))
            res["inference_reward"] = infer_true_reward

        self.fit_gp(self.x, self.y)

        return res

    def check_start_K(self, xtest, fake_xtest_size=512):
        if self.start_K is None:
            # If we are in the continuous setting, then xtest=None, but we may need an xtest for some metrics
            if xtest is None:
                bounds = np.array(self.GP.bounds)
                self.fake_xtest = torch.tensor(
                    np.random.uniform(
                        low=np.tile(bounds[:, 0], (fake_xtest_size, 1)), high=np.tile(bounds[:, 1], (fake_xtest_size, 1)), size=(fake_xtest_size, self.GP.d)
                    ),
                    dtype=torch.double,
                )
            (_, self.start_K) = self.GP.mean_var(xtest if xtest is not None else self.fake_xtest, full=True)

    def posterior_variance_stats(self, xtest, x_star, n_runs_pmax=50, eff_dim_s=False, compute_eig=False):
        (_, x_star_var) = self.GP.mean_var(x_star.view(-1, self.GP.d))
        if xtest is None:
            continuous = True
            xtest = self.fake_xtest
        else:
            continuous = False
        (_, post_K) = self.GP.mean_var(xtest, full=True)

        # reusing pmax sample computation (e.g. when DPP-TS samples from it)
        prev_samples_n = self.temp_pmax_samples.shape[0] if self.temp_pmax_samples is not None else 0
        if prev_samples_n < n_runs_pmax:
            new_n_runs_pmax = n_runs_pmax - prev_samples_n
            pmax_samples = torch.empty((new_n_runs_pmax, xtest.shape[1]), dtype=torch.double)
            for run_i in range(new_n_runs_pmax):
                (xnext, _) = self.sample_point(xtest if not continuous else None)
                pmax_samples[run_i] = xnext
            if self.temp_pmax_samples is not None:
                pmax_samples = torch.cat((self.temp_pmax_samples, pmax_samples), dim=0)
        else:
            pmax_samples = self.temp_pmax_samples

        # calculate estimated vector of Pmax probabilities for the domain
        pmax_freqs = torch.zeros(xtest.shape[0], dtype=torch.double)
        if continuous:
            distances = torch.cdist(pmax_samples, xtest)
            for pmax_s_dists in distances:
                s_idx = torch.argmin(pmax_s_dists)
                pmax_freqs[s_idx] += 1
        else:
            for pmax_s in pmax_samples:
                s_idx = (xtest == pmax_s).nonzero(as_tuple=True)
                pmax_freqs[s_idx[0]] += 1
        pmax_freqs /= pmax_samples.shape[0]

        var_diag = torch.diag(post_K)
        avg_var = torch.sum(var_diag / var_diag.shape[0])

        pmax_weighted_var = var_diag * pmax_freqs  # element-wise mult
        pmax_avg_var = torch.sum(pmax_weighted_var)

        eff_d_s = self.GP.s if eff_dim_s else 1.0
        post_eff_d = self.eff_d(post_K, eff_d_s)

        D_pmax_sqrt = torch.diag(torch.sqrt(pmax_freqs))
        pmax_start_K = D_pmax_sqrt @ self.start_K @ D_pmax_sqrt
        pmax_start_K_eff_d = self.eff_d(pmax_start_K, eff_d_s)
        pmax_post_K = D_pmax_sqrt @ post_K @ D_pmax_sqrt
        pmax_post_K_eff_d = self.eff_d(pmax_post_K, eff_d_s)

        res = {
            "x_star_var": x_star_var[0][0],
            "avg_var": avg_var,
            "pmax_avg_var": pmax_avg_var,
            "post_eff_d": post_eff_d,
            "pmax_start_K_eff_d": pmax_start_K_eff_d,
            "pmax_post_K_eff_d": pmax_post_K_eff_d,
        }

        return res
