import logging
from os import path, makedirs
from typing import Tuple

import numpy as np
import torch
from gym import Env

logger = logging.getLogger(__name__)


class System:
    def __init__(self, cache_dir: str = None, gen_params: dict = None, env: Env = None):
        self.cache_dir = cache_dir
        self.is_cached = bool(cache_dir)
        self.gen_params = gen_params or {}
        self.env = env

    def generate(self, epochs=5, steps=100, fold=4) -> Tuple[np.ndarray, ...]:
        if hasattr(self.env, 'max_episode_steps'):
            self.env.max_episode_steps = steps
        else:
            raise AttributeError('can\'t set environment\'s episode steps!!')

        data = [np.empty(0)] * 5

        for ep in range(epochs):
            obs = [self.env.reset()]
            acts = []
            for _ in range(steps):
                # TODO change to ↓ env.action_space.sample()
                acts.append(np.random.random(3))
                obs_, r, done, _ = self.env.step(acts[-1])
                obs.append(obs_)  # take a random action
                if done:
                    break

            X_, y_ = self._to_data(obs, acts)
            groups_ = np.full((X_.shape[0], 1), ep)
            X_fold_, y_fold_ = self._to_data(obs, acts, fold)
            data_ = [X_, y_, groups_, X_fold_, y_fold_]

            for i, d_ in enumerate(data_):
                data[i] = d_ if ep == 0 else np.vstack((data[i], d_))

            self.env.close()

        return tuple(data)

    @staticmethod
    def _to_data(obs, act, fold=1):
        n = len(act)
        dx, dy = len(obs[0]) + fold * len(act[0]), len(obs[0])
        X, y = np.empty((n - fold, dx)), np.empty((n - fold, dy))

        for i in range(0, n - fold):
            X[i] = np.hstack((obs[i].to_numpy(), *act[i:i + fold]))
            y[i] = obs[i].to_numpy() - obs[i + fold].to_numpy()
        return X, y

    @property
    def trajectories(self):
        tmpl = path.join(self.cache_dir, str(self.gen_params)[1:-1] + '{}.npy')
        if self.is_cached:
            data_file = tmpl.format('X')
            if path.isfile(data_file) and path.getctime(data_file) > path.getctime(__file__):
                # The data is up to date, since the data generating script is older than the data.
                logger.info('reading old data from %s', self.cache_dir)
                # TODO convert this to a proper PyTorch dataset
                X, y, groups, X_fold, y_fold = [np.load(tmpl.format(d)) for d in 'X y groups X_fold y_fold'.split()]
                X, y, X_fold, y_fold = [torch.from_numpy(d).to(dtype=torch.float32) for d in [X, y, X_fold, y_fold]]
                return X, y, groups, X_fold, y_fold
            else:
                X, y, groups, X_fold, y_fold = self.generate(**self.gen_params)
                logger.info('generating new data')
                makedirs(self.cache_dir, exist_ok=True)
                for name, v in zip('X y groups X_fold y_fold'.split(), [X, y, groups, X_fold, y_fold]):
                    np.save(tmpl.format(name), v)
        else:
            X, y, groups, X_fold, y_fold = self.generate(**self.gen_params)
        X, y, X_fold, y_fold = [torch.from_numpy(d).to(dtype=torch.float32) for d in [X, y, X_fold, y_fold]]
        return X, y, groups, X_fold, y_fold
