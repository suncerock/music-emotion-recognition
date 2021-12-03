from math import sqrt

from sklearn.metrics import mean_squared_error, r2_score


def compute_rmse(output, target):
    """
    Compute the root mean squared error between output and target
    Input:
        - output: np.ndarray, (N, )
        - target: np.ndarray, (N, )
    """
    rmse = sqrt(mean_squared_error(target, output))
    return rmse


def compute_r2(output, target):
    # def r2_score(y, x):
    #     sst = ((y - y.mean()) ** 2).sum()
    #     sse = ((x - y) ** 2).sum()
    #     return 1 - sse / sst
    r2 = r2_score(target, output)

    return r2


def compute_metric(output, target):
    output_arousal, output_valence = output['arousal'], output['valence']
    target_arousal, target_valence = target['arousal'], target['valence']
    return dict(
        r2_arousal=compute_r2(output_arousal, target_arousal),
        r2_valence=compute_r2(output_valence, target_valence),
        rmse_arousal=compute_rmse(output_arousal, target_arousal),
        rmse_valence=compute_rmse(output_valence, target_valence)
    )