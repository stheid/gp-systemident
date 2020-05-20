import logging

import gym
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_validate

from gp_systemident import System
from gp_systemident.regressors import ExactMultioutputGPR, NStateExactMultioutputGPR
from gp_systemident.regressors.variational import VariationalMultioutputGPR

logging.basicConfig(level=logging.INFO)
if __name__ == '__main__':
    env = gym.make('openmodelica_microgrid_gym:ModelicaEnv-v1',
                   viz_mode=None,
                   max_episode_steps=None,
                   model_input=['i1p1', 'i1p2', 'i1p3'],
                   model_output=dict(lc1=['inductor1.i', 'inductor2.i', 'inductor3.i']),
                   model_path='../fmu/grid.network.fmu')
    sys = System(cache_dir='data', env=env, gen_params=dict())
    X, y, groups, X_fold, y_fold = sys.trajectories

    for clf, params in [(Ridge(), {}), (VariationalMultioutputGPR(), dict(fit_params=dict(points=32))),
                        (ExactMultioutputGPR(), {})]:
        score = cross_validate(clf, X, y, groups=groups, cv=GroupKFold(), scoring=('r2', 'neg_root_mean_squared_error'),
                               return_train_score=True, **params)
        print(f'rmse={score["test_neg_root_mean_squared_error"].mean():.2f}, r2={score["test_r2"].mean():.2f}')

    cv = GroupKFold()
    train, test = next(cv.split(X, y, groups))
    reg = NStateExactMultioutputGPR(3, 3)
    reg.fit(X[train], y[train])

    # TODO use only the evaluation section of the n-fold data
    rmse = reg.score(X_fold, y_fold)
    print(f'{rmse=:.2f}')
