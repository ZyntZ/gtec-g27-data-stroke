import numpy as np

from stroke_rehab.data import Recording
from stroke_rehab.montage import (GRID, LAYOUTS, global_search, layout_fit, layout_for,
                                  mirrored)
from stroke_rehab.riemann import (RecenteredTangentSpace, filtered_trials, geodesic,
                                  riemann_mean, window_covariances)


def _spd(rng, n=4):
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def test_mean_and_geodesic_of_commuting_matrices():
    a, b = np.diag([1.0, 4.0]), np.diag([4.0, 16.0])
    assert np.allclose(riemann_mean(np.array([a, b])), np.diag([2.0, 8.0]))
    assert np.allclose(geodesic(a, b, 0.5), np.diag([2.0, 8.0]))
    assert np.allclose(geodesic(a, b, 0.0), a) and np.allclose(geodesic(a, b, 1.0), b)


def test_causal_filter_ignores_future_samples():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(4096, 2))
    changed = signal.copy()
    changed[2048 + 1024:] += 100.0  # alter EEG from four seconds into the trial
    onsets = np.array([1024])
    first = filtered_trials(Recording(signal, np.array([1]), onsets, 256), causal=True)
    second = filtered_trials(Recording(changed, np.array([1]), onsets, 256), causal=True)
    assert np.array_equal(first[:, :, :2048], second[:, :, :2048])
    offline = filtered_trials(Recording(changed, np.array([1]), onsets, 256))
    assert not np.allclose(filtered_trials(Recording(signal, np.array([1]), onsets, 256))
                           [:, :, :2048], offline[:, :, :2048])


def test_recentering_absorbs_a_gain_change_between_runs():
    rng = np.random.default_rng(1)
    def run(gain):
        y = np.tile([1, -1], 30)
        x = rng.normal(size=(60, 1, 256, 4))
        x[y == 1, 0, :, 0] *= 3.0
        x[y == -1, 0, :, 1] *= 3.0
        x[..., 2] *= gain
        return window_covariances(x, 256, (0.0, 1.0)), y
    (train, y_train), (test, y_test) = run(1.0), run(3.0)
    adaptive = RecenteredTangentSpace(C=1.0).fit(train, y_train)
    fixed = RecenteredTangentSpace(C=1.0, adapt=False).fit(train, y_train)
    assert np.mean(adaptive.predict(test) == y_test) > 0.9
    assert np.mean(fixed.predict(test) == y_test) < 0.7
    # A prediction never depends on trials that arrive after it.
    assert np.array_equal(adaptive.predict(test)[:20], adaptive.predict(test[:20]))


def test_layout_fit_prefers_the_generating_layout():
    rng = np.random.default_rng(2)
    xy = np.array([GRID[name] for name in LAYOUTS["paper"]], dtype=float)
    distance = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    correlation = np.exp(-distance / 3) + rng.normal(scale=0.01, size=distance.shape)
    correlation = (correlation + correlation.T) / 2
    paper = layout_fit(correlation, LAYOUTS["paper"], permutations=200)
    other = layout_fit(correlation, LAYOUTS["montage_png"], permutations=200)
    assert paper[0] < -0.9 and paper[0] < other[0] and paper[1] < 0.01
    assert layout_for("x/P1_post_test.mat") == LAYOUTS["montage_png"]
    assert layout_for("x/P2_post_test.mat") == LAYOUTS["paper"]
    assert layout_for("x/P1_pre_test.mat").index("C3") == 6
    assert all(sorted(layout) == sorted(set(layout)) for layout in LAYOUTS.values())


def test_global_search_recovers_a_scrambled_order_up_to_mirror():
    layout = LAYOUTS["paper"]
    xy = np.array([GRID[name] for name in layout], dtype=float)
    correlation = np.exp(-np.linalg.norm(xy[:, None] - xy[None], axis=-1) / 3)
    assert global_search(correlation, layout, restarts=6)[0] in ("stated", "mirror")
    flip = mirrored(layout)
    assert [layout[i] for i in flip][:5] == ["FC6", "FC2", "FCz", "FC1", "FC5"]
    scrambled = np.random.default_rng(3).permutation(16)
    assert global_search(correlation[np.ix_(scrambled, scrambled)], layout, restarts=6)[0] == "other"
