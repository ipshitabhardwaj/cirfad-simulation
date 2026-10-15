"""
CIRFAD simulation study.

Synthetic primary/reference accelerometer signals for one monitored machine (M0)
sharing a floor with three intermittent interferers. Evaluates:
  1. Interference rejection of LMS vs NLMS (dB), including reference leakage.
  2. Window-level false-positive and detection rates for four pipeline variants:
     raw/static, raw/context, NLMS/static, NLMS/context.
All signals are in units of g. fs = 1 kHz (MPU-6050 accelerometer output-rate limit).
"""
import json, sys
import numpy as np
from numba import njit
from scipy import signal

FS = 1000
WIN = FS                      # 1 s windows
SENSOR_NOISE = 0.0063         # g RMS, ~400 ug/sqrt(Hz) over ~250 Hz
B_LP, A_LP = signal.butter(4, 250, fs=FS)   # stands in for the MPU-6050 DLPF

# ----------------------------------------------------------------- adaptive filters
@njit(cache=True)
def lms(xp, xr, N, mu):
    w = np.zeros(N); e = np.empty_like(xp); buf = np.zeros(N)
    for k in range(xp.size):
        for j in range(N - 1, 0, -1):
            buf[j] = buf[j - 1]
        buf[0] = xr[k]
        y = 0.0
        for j in range(N):
            y += w[j] * buf[j]
        ek = xp[k] - y
        e[k] = ek
        for j in range(N):
            w[j] += 2.0 * mu * ek * buf[j]
    return e

@njit(cache=True)
def nlms(xp, xr, N, mu, eps):
    w = np.zeros(N); e = np.empty_like(xp); buf = np.zeros(N); p = 0.0
    for k in range(xp.size):
        old = buf[N - 1]
        for j in range(N - 1, 0, -1):
            buf[j] = buf[j - 1]
        buf[0] = xr[k]
        p += xr[k] * xr[k] - old * old       # running ||x_r||^2
        if p < 0.0:
            p = 0.0
        y = 0.0
        for j in range(N):
            y += w[j] * buf[j]
        ek = xp[k] - y
        e[k] = ek
        g = mu * ek / (eps + p)
        for j in range(N):
            w[j] += g * buf[j]
    return e

@njit(cache=True)
def nlms_gated(xp, xr, N, mu, eps, gate_ratio, blk):
    """NLMS that adapts only while short-term reference power exceeds gate_ratio times
    its slowly tracked quiet floor; weights are frozen otherwise (limits signal cancellation
    through reference leakage when neighbouring machines are idle)."""
    w = np.zeros(N); e = np.empty_like(xp); buf = np.zeros(N); p = 0.0
    floor = -1.0; blkp = 0.0; adapt = False; nb = 0
    for k in range(xp.size):
        old = buf[N - 1]
        for j in range(N - 1, 0, -1):
            buf[j] = buf[j - 1]
        buf[0] = xr[k]
        p += xr[k] * xr[k] - old * old
        if p < 0.0:
            p = 0.0
        blkp += xr[k] * xr[k]; nb += 1
        if nb == blk:
            bp = blkp / blk
            if floor < 0.0:
                floor = bp
            elif bp < floor:
                floor = bp                       # fast attack downward
            else:
                floor += 1e-4 * (bp - floor)     # very slow upward drift
            adapt = bp > gate_ratio * floor
            blkp = 0.0; nb = 0
        y = 0.0
        for j in range(N):
            y += w[j] * buf[j]
        ek = xp[k] - y
        e[k] = ek
        if adapt:
            g = mu * ek / (eps + p)
            for j in range(N):
                w[j] += g * buf[j]
    return e

# ----------------------------------------------------------------- schedules
def piecewise(rng, n_sec, levels, min_s, max_s, probs=None):
    """Piecewise-constant per-second schedule."""
    out = np.empty(n_sec); t = 0
    while t < n_sec:
        d = rng.integers(min_s, max_s + 1)
        out[t:t + d] = rng.choice(levels, p=probs)
        t += d
    return out

def to_samples(per_sec, ramp_s=1.0):
    x = np.repeat(per_sec, FS)
    k = int(ramp_s * FS)
    return np.convolve(x, np.ones(k) / k, mode="same")

def bandnoise(rng, n, lo, hi):
    b, a = signal.butter(2, [lo, hi], btype="band", fs=FS)
    x = signal.lfilter(b, a, rng.standard_normal(n))
    return x / x.std()

def resonance(fn, decay, n=120):
    t = np.arange(n) / FS
    return np.exp(-decay * t) * np.sin(2 * np.pi * fn * t)

def impulse_train(rng, inst_rate, jitter=0.01):
    """Impulses at times where cumulative event phase crosses an integer."""
    ph = np.cumsum(inst_rate) / FS
    idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0]
    j = max(int(jitter * FS / inst_rate.mean()), 0)
    idx = idx + rng.integers(-j, j + 1, idx.size)
    idx = idx[(idx > 0) & (idx < inst_rate.size)]
    x = np.zeros(inst_rate.size); x[idx] = 1.0
    return x

def delay_gain(x, d, g):
    y = np.zeros_like(x); y[d:] = g * x[:-d] if d > 0 else g * x
    return y

# ----------------------------------------------------------------- scenario
PATHS = [(2, 1.0, 3), (3, 0.9, 5), (1, 1.1, 2)]   # (delay to reference, gain, extra delay to primary)

def scenario(seed, n_sec, fault=None, severity=0.0, leak=0.10):
    rng = np.random.default_rng(seed)
    n = n_sec * FS
    t = np.arange(n) / FS

    # operating context
    load_s = piecewise(rng, n_sec, [0.3, 0.6, 0.9], 300, 1200)
    load_s = np.clip(load_s + 0.03 * rng.standard_normal(n_sec), 0.1, 1.0)
    temp_s = np.empty(n_sec); T = 32.0
    for i in range(n_sec):          # first-order thermal lag toward load-dependent target
        T += (30 + 18 * load_s[i] - T) / 900.0
        temp_s[i] = T
    temp_s += 0.3 * rng.standard_normal(n_sec)
    current_s = 4.0 + 18.0 * load_s + 0.4 * rng.standard_normal(n_sec)   # A, rated ~22 A
    load = to_samples(load_s, 2.0)
    temp = np.repeat(temp_s, FS)

    # ---- target machine M0: 4-pole induction motor, slip grows with load
    fr = 24.9 - 1.0 * load
    ph = 2 * np.pi * np.cumsum(fr) / FS
    a1 = 0.020 + 0.020 * load
    s = a1 * np.sin(ph + 0.3) + 0.35 * a1 * np.sin(2 * ph + 1.1)
    s += (0.006 + 0.006 * load) * (1 + 0.01 * (temp - 38)) * bandnoise(rng, n, 15, 300)
    if fault == "imbalance":
        s += severity * a1 * np.sin(ph + 0.3)
    elif fault == "looseness":
        for h in range(3, 8):
            s += severity * a1 * 0.5 ** (h - 3) * np.sin(h * ph + rng.uniform(0, 6.28))
        s += severity * 0.3 * a1 * np.sin(0.5 * ph)
    elif fault == "bearing_outer":
        bpfo = 3.585 * fr
        imp = impulse_train(rng, bpfo)
        imp *= 1 + 0.2 * rng.standard_normal(n)
        s += severity * signal.lfilter(resonance(185, 220), [1], imp)

    # ---- interferers (sources, before propagation)
    on1 = to_samples(piecewise(rng, n_sec, [0, 1], 120, 900), 0.5)   # reciprocating compressor
    on2 = to_samples(piecewise(rng, n_sec, [0, 1], 120, 900), 0.5)   # neighbouring motor
    on3 = to_samples(piecewise(rng, n_sec, [0, 1], 120, 900, [0.6, 0.4]), 0.2)  # press

    f1 = 12.4 + 0.05 * np.sin(2 * np.pi * t / 37)
    ph1 = 2 * np.pi * np.cumsum(f1) / FS
    i1 = sum(0.030 / h ** 0.8 * np.sin(h * ph1 + h) for h in range(1, 10))
    i1 += 0.006 * bandnoise(rng, n, 40, 200)
    f2 = 24.6 + 0.08 * np.sin(2 * np.pi * t / 53)          # close to M0's 1x
    ph2 = 2 * np.pi * np.cumsum(f2) / FS
    i2 = 0.035 * np.sin(ph2) + 0.012 * np.sin(2 * ph2 + 0.4) + 0.008 * bandnoise(rng, n, 20, 250)
    rate3 = np.full(n, 1.2)
    i3 = 0.25 * signal.lfilter(resonance(95, 60, 200), [1], impulse_train(rng, rate3, 0.05))
    srcs = [i1 * on1, i2 * on2, i3 * on3]

    # propagation: the reference sits on the floor near M0's base. Floor vibration reaching
    # M0's bearing housing passes through M0's own base/mount (common to all sources) and
    # arrives a few ms later than at the reference; the extra delay depends on the direction
    # of each source, so one FIR cannot match all three paths exactly.
    b_m, a_m = signal.iirpeak(140, 3, fs=FS)
    n_p = np.zeros(n); n_r = np.zeros(n)
    for src, (dr, gr, dd) in zip(srcs, PATHS):
        fl = delay_gain(src, dr, gr)
        n_r += fl
        y = delay_gain(fl, dd, 0.8)
        n_p += 0.8 * y + 0.25 * signal.lfilter(b_m, a_m, y)

    # leakage of the machine's own vibration into the reference
    s_leak = delay_gain(s, 4, leak)

    xp_clean = s + n_p
    xp = signal.lfilter(B_LP, A_LP, xp_clean + SENSOR_NOISE * rng.standard_normal(n))
    xr = signal.lfilter(B_LP, A_LP, n_r + s_leak + SENSOR_NOISE * rng.standard_normal(n))
    s_obs = signal.lfilter(B_LP, A_LP, s)
    np_obs = signal.lfilter(B_LP, A_LP, n_p)
    interf_on = (on1[::FS] > 0.5) | (on2[::FS] > 0.5) | (on3[::FS] > 0.5)
    return dict(xp=xp, xr=xr, s=s_obs, n=np_obs, current=current_s, temp=temp_s,
                load=load_s, interf_on=interf_on)

# ----------------------------------------------------------------- features
def features(x):
    w = x[: x.size // WIN * WIN].reshape(-1, WIN)
    w = w - w.mean(axis=1, keepdims=True)
    rms = np.sqrt((w ** 2).mean(axis=1))
    kurt = (w ** 4).mean(axis=1) / rms ** 4
    cf = np.abs(w).max(axis=1) / rms
    X = np.abs(np.fft.rfft(w * np.hanning(WIN), axis=1))
    f = np.fft.rfftfreq(WIN, 1 / FS)
    sc = (X * f).sum(axis=1) / X.sum(axis=1)
    return np.stack([rms, kurt, cf, sc], axis=1)

# ----------------------------------------------------------------- context thresholds
LOAD_EDGES = (0.45, 0.75)          # fraction of rated current above idle
def load_class(current):
    frac = (current - 4.0) / 18.0
    return np.digitize(frac, LOAD_EDGES)

class Thresholds:
    """Robust per-context baselines (median, scaled MAD) with hierarchical back-off.
    ctx: 'static' | 'load_temp' | 'load_ref' | 'load_temp_ref'."""
    MIN_CELL = 300

    def fit(self, F, current, temp, refrms, ctx="load_temp"):
        self.ctx = ctx
        self.t_split = np.median(temp)
        self.r_split = 2.0 * np.percentile(refrms, 10)
        cells = self._cells(current, temp, refrms)
        lc = load_class(current)
        self.glob = self._stats(F)
        self.by_load = {l: self._stats(F[lc == l]) for l in range(3) if (lc == l).sum() >= self.MIN_CELL}
        self.by_cell = {c: self._stats(F[cells == c]) for c in np.unique(cells)
                        if (cells == c).sum() >= self.MIN_CELL}
        return self

    def _cells(self, current, temp, refrms):
        lc = load_class(current)
        tb = (temp > self.t_split).astype(int)
        rb = (refrms > self.r_split).astype(int)
        if self.ctx == "load_temp":
            return lc * 2 + tb
        if self.ctx == "load_ref":
            return lc * 2 + rb
        if self.ctx == "load_temp_ref":
            return lc * 4 + tb * 2 + rb
        return np.zeros_like(lc)

    @staticmethod
    def _stats(F):
        med = np.median(F, axis=0)
        mad = 1.4826 * np.median(np.abs(F - med), axis=0)
        return med, np.maximum(mad, 1e-9)

    def z(self, F, current, temp, refrms):
        Z = np.empty_like(F)
        lc = load_class(current); cells = self._cells(current, temp, refrms)
        for i in range(F.shape[0]):
            if self.ctx == "static":
                med, mad = self.glob
            elif cells[i] in self.by_cell:
                med, mad = self.by_cell[cells[i]]
            elif lc[i] in self.by_load:
                med, mad = self.by_load[lc[i]]
            else:
                med, mad = self.glob
            Z[i] = (F[i] - med) / mad
        Z[:, 3] = np.abs(Z[:, 3])          # spectral centroid: two-sided
        return Z

def refrms(sc):
    x = sc["xr"]; w = x[: x.size // WIN * WIN].reshape(-1, WIN)
    return w.std(axis=1)

def alerts(Z, z_caution=3.0, persist=5):
    score = np.clip(20.0 * Z, 0, 100).max(axis=1)      # A = max per-feature score
    raw = score >= 20.0 * z_caution
    # persistence: alert only after `persist` consecutive raw exceedances
    out = np.zeros_like(raw); run = 0
    for i, r in enumerate(raw):
        run = run + 1 if r else 0
        out[i] = run >= persist
    return out, score

# ----------------------------------------------------------------- experiments
MU_NLMS, EPS, GATE, BLK = 0.0005, 1e-3, 2.0, 250

def clean(sc, method, N=32, mu=None):
    if method == "raw":
        return sc["xp"]
    if method == "lms":                         # original design: plain LMS, mu = 0.005
        return lms(sc["xp"], sc["xr"], N, 0.005 if mu is None else mu)
    if method == "nlms":                        # NLMS without gating
        return nlms(sc["xp"], sc["xr"], N, MU_NLMS if mu is None else mu, EPS)
    if method == "gnlms":                       # proposed: gated NLMS
        return nlms_gated(sc["xp"], sc["xr"], N, MU_NLMS if mu is None else mu, EPS, GATE, BLK)
    if method == "oracle":                      # interference-free bound (not realisable)
        return sc["s"]
    raise ValueError(method)

def rejection_db(sc, e, skip_s=60):
    k = skip_s * FS
    resid = e[k:] - sc["s"][k:]
    return 10 * np.log10((sc["n"][k:] ** 2).mean() / (resid ** 2).mean())

def wiener_ceiling(sc, N=32, skip_s=60):
    k = skip_s * FS
    on = np.repeat(sc["interf_on"], FS)[k:]
    xr = sc["xr"]
    X = np.stack([np.roll(xr, i) for i in range(N)], 1)[k:][on]
    y = sc["xp"][k:][on]
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ w - sc["s"][k:][on]
    return 10 * np.log10(sc["n"][k:][on].var() / r.var())

if __name__ == "__main__":
    out = {}
    BASE_S, TEST_S, FAULT_S = 3 * 3600, 3 * 3600, 2400

    # ---------- 1. interference rejection vs reference leakage (3 seeds, 30 min each)
    rej = {}
    for leak in [0.0, 0.05, 0.10, 0.20, 0.30]:
        acc = {m: [] for m in ["lms", "nlms", "gnlms", "wiener"]}
        for seed in [11, 12, 13]:
            sc = scenario(seed, 1800, leak=leak)
            for m in ["lms", "nlms", "gnlms"]:
                acc[m].append(rejection_db(sc, clean(sc, m)))
            acc["wiener"].append(wiener_ceiling(sc))
        rej[leak] = {m: [float(np.mean(v)), float(np.std(v))] for m, v in acc.items()}
        print("leak", leak, rej[leak], flush=True)
    out["rejection_vs_leak"] = rej

    # filter length / step size sweep for gated NLMS at leak 0.1
    sc = scenario(11, 1800, leak=0.10)
    out["sweep"] = {}
    for N in [8, 16, 32, 64]:
        for mu in [0.001, 0.0025, 0.005, 0.01, 0.02]:
            out["sweep"][f"N{N}_mu{mu}"] = float(rejection_db(sc, clean(sc, "lms", N, mu)))
    print("sweep", out["sweep"], flush=True)

    # traces for figures
    sc = scenario(5, 1800, leak=0.10)
    np.savez_compressed("conv.npz", xp=sc["xp"], s=sc["s"], n=sc["n"],
                        e_lms=clean(sc, "lms"), e_nlms=clean(sc, "nlms"), e_g=clean(sc, "gnlms"),
                        on=sc["interf_on"])

    # ---------- 2. detection study: 5 independent repetitions
    faults = [("imbalance", 0.6), ("imbalance", 1.0),
              ("looseness", 0.5), ("looseness", 1.0),
              ("bearing_outer", 0.04), ("bearing_outer", 0.08)]
    configs = [("raw", "static"), ("raw", "load_temp"), ("lms", "static"),
               ("lms", "load_temp"), ("oracle", "load_temp")]
    reps = []
    for rep in range(5):
        base = scenario(1000 + rep, BASE_S, leak=0.10)
        test = scenario(2000 + rep, TEST_S, leak=0.10)
        fsc = {k: scenario(3000 + 10 * rep + i, FAULT_S, *k, leak=0.10) for i, k in enumerate(faults)}
        rb, rt = refrms(base), refrms(test)
        rf = {k: refrms(v) for k, v in fsc.items()}
        res = {}
        for method in ["raw", "lms", "oracle"]:
            Fb = features(clean(base, method)); Ft = features(clean(test, method))
            Ff = {k: features(clean(v, method)) for k, v in fsc.items()}
            for m2, ctx in configs:
                if m2 != method:
                    continue
                th = Thresholds().fit(Fb, base["current"], base["temp"], rb, ctx)
                a_t, _ = alerts(th.z(Ft, test["current"], test["temp"], rt))
                on = test["interf_on"][: a_t.size]
                r = {"fpr": float(a_t.mean()), "fpr_on": float(a_t[on].mean()),
                     "fpr_off": float(a_t[~on].mean())}
                for k, F in Ff.items():
                    a_f, _ = alerts(th.z(F, fsc[k]["current"], fsc[k]["temp"], rf[k]))
                    r[f"{k[0]}@{k[1]}"] = float(a_f.mean())
                res[f"{method}|{ctx}"] = r
        reps.append(res)
        print("rep", rep, json.dumps(res), flush=True)
    out["detection_reps"] = reps
    json.dump(out, open("results.json", "w"), indent=2)
