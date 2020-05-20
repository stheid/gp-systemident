import logging

import gym
from sklearn.model_selection import cross_val_score, GroupKFold

from gp_systemident import System
from gp_systemident.regressors import ExactMultioutputGPR, NStateExactMultioutputGPR

logging.basicConfig(level=logging.INFO)
if __name__ == '__main__':
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv-v1',
                   viz_mode=None,
                   model_input=['i1p1', 'i1p2', 'i1p3'],
                   model_output=dict(lc1=['inductor1.i', 'inductor2.i', 'inductor3.i']),
                   model_path='../fmu/grid.network.fmu')
    sys = System(cache_dir='data', env=env)
    X, y, groups, X_fold, y_fold = sys.trajectories

    rmse = cross_val_score(ExactMultioutputGPR(), X, y, groups=groups, cv=GroupKFold()).mean()
    print(f'rmse={rmse:.2f}')

    cv = GroupKFold()
    train, test = next(cv.split(X, y, groups))
    reg = NStateExactMultioutputGPR(3, 3)
    reg.fit(X[train], y[train])
    rmse = reg.score(X_fold, y_fold)

    print(f'rmse={rmse:.2f}')
