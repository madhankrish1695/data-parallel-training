
### Weak scaling

| N | global batch | epoch time (s) | speedup | efficiency (%) | total time (s) | final acc (%) | acc gap | comm/step (ms) | exposed comm (% of step) | raw all-reduce (ms) | data wait (% epoch) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 52.67 | 1.00 | 100.0 | 1581 | 93.50 | +0.00 | 3.5 | 2.4 | 0.1 | 0.3 |
| 2 | 256 | 27.11 | 1.94 | 97.1 | 814 | 93.96 | -0.46 | 7.0 | 5.0 | 11.1 | 1.0 |

### Strong scaling

| N | global batch | epoch time (s) | speedup | efficiency (%) | total time (s) | final acc (%) | acc gap | comm/step (ms) | exposed comm (% of step) | raw all-reduce (ms) | data wait (% epoch) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 512 | 49.56 | 1.00 | 100.0 | 1487 | 93.67 | +0.00 | -21.0 | 0.0 | 0.1 | 0.7 |
| 2 | 512 | 26.74 | 1.85 | 92.7 | 802 | 93.49 | +0.18 | 8.6 | 2.9 | 11.1 | 1.6 |

### Time to 90% test accuracy (s)

| run | time (s) |
|---|---|
| strong_N1 | 1090.919301721001 |
| strong_N2 | 588.5346268000003 |
| weak_N1 | 1001.7611718989992 |
| weak_N2 | 570.0606857649991 |
| weak_amp_N2 | 246.6224248219978 |