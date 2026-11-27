# Validation

The reference runs used RTX 4090 GPUs with Python 3.11, PyTorch 2.5.1, and Lightning 2.5.5.

## Reference full Gibson evaluation

Both checkpoints were evaluated on 2,007 Gibson-f test frames and 37 Gibson-t trajectories of 100 frames, using 2,000 sampled locations per frame and a fixed seed (42). No precomputed distance fields were used.

Sequential evaluation skips the first three frames of each trajectory, leaving 3,589 evaluated frames. It uses relative motion from recorded poses and reports success and RMSE over the final ten frames of each trajectory.

| Checkpoint | 0.1 m recall | 0.5 m recall | 1 m recall | 1 m / 30° recall | 1 m seq. success | RMSE succ. | RMSE all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `freeloc_f3loc_depth.ckpt` | 9.92% | 39.36% | 44.64% | 42.85% | 100.00% |  0.15 m | 0.15 m |
| `freeloc_unloc_depth.ckpt` | 16.59% | 46.94% | 50.22% | 48.73% | 100.00% | 0.15 m | 0.15 m |
