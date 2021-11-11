from math import sqrt

from sklearn.metrics import mean_squared_error , r2_score


def compute_rmse(output, target, music_id):
    """
    Compute the root mean squared error between output and target
    Input:
        - output: np.ndarray, (N, )
        - target: np.ndarray, (N, )
        - music_id: np.ndarray, (N, )
    """
    rmse_across_segments = sqrt(mean_squared_error(target, output))

    rmse_across_songs = 0
    music_id_set = set(music_id)
    for music in music_id_set:
        rmse_across_songs += sqrt(mean_squared_error(target[music_id == music], output[music_id == music]))
    rmse_across_songs /= len(music_id_set)
    
    return rmse_across_segments, rmse_across_songs


def compute_r2(output, target, music_id):
    # def r2_score(y, x):
    #     sst = ((y - y.mean()) ** 2).sum()
    #     sse = ((x - y) ** 2).sum()
    #     return 1 - sse / sst
    r2_across_segments = r2_score(target, output)

    r2_across_songs = 0
    music_id_set = set(music_id)
    for music in music_id_set:
        r2_across_songs += r2_score(target[music_id == music], output[music_id == music])
    r2_across_songs /= len(music_id_set)
    return r2_across_segments, r2_across_songs


def compute_metric(output, target, music_id):
    output_arousal, output_valence = output['arousal'], output['valence']
    target_arousal, target_valence = target['arousal'], target['valence']

    rmse_segments_arousal, rmse_songs_arousal = compute_rmse(output_arousal, target_arousal, music_id)
    r2_segments_arousal, r2_songs_arousal = compute_r2(output_arousal, target_arousal, music_id)
    rmse_segments_valence, rmse_songs_valence = compute_rmse(output_valence, target_valence, music_id)
    r2_segments_valence, r2_songs_valence = compute_r2(output_valence, target_valence, music_id)
    return dict(
        rmse_segments_arousal=rmse_segments_arousal,
        rmse_songs_arousal=rmse_songs_arousal,
        r2_segments_arousal=r2_segments_arousal,
        r2_songs_arousal=r2_songs_arousal,
        rmse_segments_valence=rmse_segments_valence,
        rmse_songs_valence=rmse_songs_valence,
        r2_segments_valence=r2_segments_valence,
        r2_songs_valence=r2_songs_valence
    )
