# CIRFAD simulation

Simulation code for the paper **"CIRFAD: Low-Cost Retrofit Vibration Monitoring with Reference-Sensor Interference Cancellation and Load-Aware Baselines for Multi-Machine MSME Workshops"** (Ipshita Bhardwaj, Shradhanjali, Kamaldeep Singh; Chandigarh Engineering College, CGC Landran).

It reproduces every number in Section 6 of the paper (Tables 5 and 6) and Figures 1 to 4.

> All results here come from a synthetic model, not from hardware. They compare pipeline variants under identical conditions and are not a prediction of field performance. See Section 6.4 of the paper.

## What it simulates

One monitored machine (M0) sharing a floor with three neighbouring machines that switch on and off. A primary accelerometer sits on the bearing housing of M0 and a reference accelerometer sits on the floor near its footing.

| Element | Model |
|---|---|
| Monitored machine | Four-pole induction motor, speed and vibration amplitude vary with load; frame temperature follows load with a lag |
| Interferers | Reciprocating compressor (12.4 Hz with harmonics), motor at 24.6 Hz (within 1 Hz of M0), press with impacts exciting a 95 Hz floor resonance |
| Paths | Floor vibration reaches the reference 1 to 3 ms before it reaches the bearing housing, through a mount response with a 140 Hz resonance |
| Leakage | Part of M0's own vibration reaches the reference (gain 0.1 by default) |
| Sensors | 1 kHz sampling, 6.3 mg RMS noise, 250 Hz low-pass (MPU-6050-like) |
| Faults | Imbalance, looseness and outer-race bearing defect, two severities each |

The pipeline: LMS canceller (32 taps, mu = 0.005), four features per 1 s window (RMS, kurtosis, crest factor, spectral centroid), median and MAD baselines indexed by load class and temperature band, and a 5-window persistence alert.

## Requirements

Python 3.10 or newer.

```
pip install -r requirements.txt
```

Tested with Python 3.10 (Windows) and 3.13 (Linux); both runs gave the same results.

## Running

```
python sim.py     # about 8 to 10 min on 2 cores; first run is slower while Numba compiles
python figs.py    # needs results.json and conv.npz from sim.py
```

`sim.py` prints progress (five `leak` lines, one `sweep` line, then `rep 0` to `rep 4`) and writes:

- `results.json`: all numbers used in the paper
- `conv.npz`: signal traces for Figure 2 (about 80 MB, not committed)

`figs.py` writes `fig1_architecture.png` to `fig4_detection.png`.

All scenarios use fixed random seeds, so a run should match the committed `results.json` up to floating-point rounding.

## Where each result comes from

| Paper item | Source |
|---|---|
| Table 5, Fig. 3 (interference rejection vs leakage) | `results.json` key `rejection_vs_leak` |
| N and mu sweep (Section 6.2 text) | `results.json` key `sweep` |
| Table 6, Fig. 4 (false positives and detection) | `results.json` key `detection_reps` (5 repetitions; mean and std are taken across them) |
| Fig. 2 (residual interference over time) | `conv.npz` |
| Fig. 1 (architecture) | drawn in `figs.py` |

In `detection_reps`, pipeline keys are `raw|static`, `raw|load_temp`, `lms|static`, `lms|load_temp` (CIRFAD) and `oracle|load_temp` (interference-free reference that observes the machine signal directly; not realisable).

## Changing the experiment

Main settings in `sim.py`:

- `BASE_S, TEST_S, FAULT_S`: commissioning, healthy-test and per-fault durations in seconds. Reduce for a quick check.
- `leak` values in the first loop, and `leak=0.10` in the detection study.
- `PATHS`: delay to reference, gain, extra delay to primary for each interferer.
- `faults`: fault types and severities.
- `alerts(Z, z_caution=3.0, persist=5)`: alert threshold and persistence.
- `MU_NLMS, EPS, GATE, BLK`: settings for the NLMS variants used for comparison.

Changing any of these changes the numbers, so they will no longer match the paper.

## Files

| File | Purpose |
|---|---|
| `sim.py` | Signal model, adaptive filters, features, baselines, experiments |
| `figs.py` | Figures 1 to 4 |
| `results.json` | Output of the run used in the paper |
| `fig*.png` | Figures as used in the paper |

## Citation

If you use this code, please cite the paper (citation to be added on publication).
