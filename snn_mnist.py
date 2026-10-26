"""
snn_mnist.py  --  Feed-forward spiking network (Brian2) for Poisson-encoded MNIST.

Pipeline
--------
  28x28 grey image  -> Poisson spike trains (rate = intensity * MAX_RATE) for T_ENC
                    -> hidden layer of LIF neurons (fixed random excitatory weights)
                    -> spike counts per hidden neuron  (feature vector)
                    -> linear read-out (sklearn LogisticRegression), trained offline.

Implementation notes
--------------------
* Images are simulated in *batches*: B independent copies of the network live in
  one Brian2 Network (block-diagonal connectivity).  Input rates for the whole
  dataset are stored in a 2-D TimedArray with dt = T_ENC, so a single `run()`
  call streams all images through the network.  At every image boundary the
  membrane potential of the hidden neurons is reset (run_regularly), so the
  images are independent.
* Hidden spikes are binned per (image, neuron, 10-ms time-bin) so we can later
  study accuracy as a function of the decision window (latency-to-decision).
"""
import gzip
import os
import struct
import time

import numpy as np
from brian2 import (Hz, Network, NeuronGroup, PoissonGroup, SpikeMonitor,
                    StateMonitor, Synapses, TimedArray, defaultclock, ms, mV,
                    prefs, second, volt)

prefs.codegen.target = "numpy"      # no compiler needed on Windows


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def _read_idx(path):
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        ndim = magic & 0xFF
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist(root="data"):
    """Returns X_train (60000, 784) in [0,1], y_train, X_test (10000, 784), y_test."""
    Xtr = _read_idx(os.path.join(root, "train-images-idx3-ubyte.gz")).reshape(-1, 784) / 255.0
    ytr = _read_idx(os.path.join(root, "train-labels-idx1-ubyte.gz")).astype(int)
    Xte = _read_idx(os.path.join(root, "t10k-images-idx3-ubyte.gz")).reshape(-1, 784) / 255.0
    yte = _read_idx(os.path.join(root, "t10k-labels-idx1-ubyte.gz")).astype(int)
    return Xtr.astype(np.float64), ytr, Xte.astype(np.float64), yte


# ----------------------------------------------------------------------------
# Default parameters
# ----------------------------------------------------------------------------
P = dict(
    n_in=784,
    n_hidden=200,
    t_enc=100 * ms,          # encoding window per image
    max_rate=100 * Hz,       # pixel intensity 1.0 -> 100 Hz
    tau_m=20 * ms,
    v_rest=-70 * mV,
    v_reset=-70 * mV,
    v_th=-50 * mV,
    t_ref=2 * ms,
    w_mean=0.5,              # weights ~ N(0.5, 0.1), clipped >= 0 (dimensionless)
    w_std=0.1,
    w_scale=0.25 * mV,       # EPSP jump per unit weight
    p_conn=1.0,              # connection probability (1.0 = dense)
    batch=25,                # images simulated in parallel
    dt=0.1 * ms,
    n_bins=10,               # time bins per image (for latency analysis)
)


def make_weights(p, seed=0):
    """Random excitatory weight matrix W (n_in x n_hidden) and the 0/1 mask."""
    rng = np.random.default_rng(seed)
    W = rng.normal(p["w_mean"], p["w_std"], size=(p["n_in"], p["n_hidden"]))
    W = np.clip(W, 0.0, None)
    mask = rng.random((p["n_in"], p["n_hidden"])) < p["p_conn"]
    return W * mask, mask


# ----------------------------------------------------------------------------
# Network construction
# ----------------------------------------------------------------------------
LIF_EQS = """
dv/dt = (v_rest - v) / tau_m : volt (unless refractory)
theta : volt   (constant)
"""


def build_network(rates_vals, W, p, theta_frozen=None, record_v=None, homeostasis=None):
    """
    rates_vals : (n_chunks, B*n_in) array of Poisson rates (Hz, plain floats)
    W          : (n_in, n_hidden) weights (dimensionless, multiplied by w_scale)
    theta_frozen : optional per-neuron threshold array (volt); default p['v_th']
    homeostasis: None or dict(r_target=..Hz, tau_r=..s, eta=..volt, theta_min, theta_max)
                 -> threshold becomes a dynamic variable driven to a target rate
    """
    B, n_in, n_h = p["batch"], p["n_in"], p["n_hidden"]
    stim = TimedArray(rates_vals * Hz, dt=p["t_enc"])
    inp = PoissonGroup(B * n_in, rates="stim(t, i)", namespace={"stim": stim}, name="inp")

    ns = dict(stim=stim, v_rest=p["v_rest"], tau_m=p["tau_m"])
    if homeostasis is None:
        eqs = LIF_EQS
    else:
        # r      : low-pass estimate of the neuron's own firing rate
        # theta  : adaptive threshold, integrates (r - r_target)
        eqs = """
        dv/dt = (v_rest - v) / tau_m : volt (unless refractory)
        dr/dt = -r / tau_r : Hz
        dtheta/dt = eta * (r - r_target) : volt
        """
        ns.update(tau_r=homeostasis["tau_r"], eta=homeostasis["eta"],
                  r_target=homeostasis["r_target"],
                  theta_min=homeostasis["theta_min"], theta_max=homeostasis["theta_max"])
    hid = NeuronGroup(B * n_h, eqs, threshold="v > theta",
                      reset="v = v_reset" if homeostasis is None else "v = v_reset; r += 1/tau_r",
                      refractory=p["t_ref"], method="euler", namespace={**ns, "v_reset": p["v_reset"]},
                      name="hid")
    hid.v = p["v_rest"]
    hid.theta = p["v_th"] if theta_frozen is None else np.tile(theta_frozen, B)
    # reset membrane at every image boundary so images are independent
    hid.run_regularly("v = v_rest", dt=p["t_enc"], when="start")
    if homeostasis is not None:
        hid.run_regularly("theta = clip(theta, theta_min, theta_max)", dt=1 * ms)

    syn = Synapses(inp, hid, "w : volt", on_pre="v_post += w", name="syn")
    pre, post = np.nonzero(W)
    i_all = np.concatenate([pre + b * n_in for b in range(B)])
    j_all = np.concatenate([post + b * n_h for b in range(B)])
    syn.connect(i=i_all, j=j_all)
    syn.w = np.tile(W[pre, post], B) * p["w_scale"]

    mon = SpikeMonitor(hid, name="mon_hid")
    objs = [inp, hid, syn, mon]
    if record_v is not None:
        objs.append(StateMonitor(hid, "v", record=record_v, name="mon_v"))
        objs.append(SpikeMonitor(inp, name="mon_in"))
    net = Network(objs)
    return net


def simulate(X, W, p, theta_frozen=None, verbose=True):
    """
    Stream all images in X (n_img, 784) through the network.
    Returns counts_t : (n_img, n_hidden, n_bins) spike counts per time bin.
    """
    defaultclock.dt = p["dt"]
    B, n_in, n_h = p["batch"], p["n_in"], p["n_hidden"]
    n_img = X.shape[0]
    n_chunks = int(np.ceil(n_img / B))
    Xpad = np.zeros((n_chunks * B, n_in))
    Xpad[:n_img] = X
    rates_vals = Xpad.reshape(n_chunks, B * n_in) * float(p["max_rate"] / Hz)

    net = build_network(rates_vals, W, p, theta_frozen=theta_frozen)
    t0 = time.time()
    net.run(n_chunks * p["t_enc"])
    wall = time.time() - t0

    mon = net["mon_hid"]
    t = np.asarray(mon.t / p["t_enc"])          # time in units of t_enc
    k = np.floor(t).astype(int)                 # chunk index
    frac = t - k
    tb = np.minimum((frac * p["n_bins"]).astype(int), p["n_bins"] - 1)
    idx = np.asarray(mon.i)
    b = idx // n_h
    h = idx % n_h
    img = k * B + b
    flat = (img * n_h + h) * p["n_bins"] + tb
    counts_t = np.bincount(flat, minlength=n_chunks * B * n_h * p["n_bins"])
    counts_t = counts_t.reshape(n_chunks * B, n_h, p["n_bins"])[:n_img]
    if verbose:
        rate = counts_t.sum() / (n_img * n_h * float(p["t_enc"] / second))
        print(f"  simulated {n_img} images in {wall:6.1f}s  |  mean hidden rate = {rate:5.1f} Hz  "
              f"|  spikes/image = {counts_t.sum() / n_img:7.1f}")
    return counts_t


def calibrate_homeostasis(X_calib, W, p, r_target=20 * Hz, tau_r=0.5 * second,
                          eta=0.5 * mV, theta_min=None, theta_max=None, sim_time=20 * second,
                          verbose=True):
    """
    Run the network with an adaptive threshold on a calibration stream and
    return the per-neuron thresholds reached at the end (then frozen).
    Threshold dynamics:  d theta/dt = eta * (r - r_target), r = low-passed own rate.
    The B parallel replicas each see a different image stream; the calibration
    images are cycled until every replica has adapted for `sim_time` of
    simulated time (>> tau_r, so the thresholds converge).
    Returns the frozen thresholds and a (n_chunks, n_hidden) trace of replica 0.
    """
    defaultclock.dt = p["dt"]
    B, n_in, n_h = p["batch"], p["n_in"], p["n_hidden"]
    n_needed = int(round(float(sim_time / p["t_enc"]))) * B
    reps = int(np.ceil(n_needed / X_calib.shape[0]))
    X_calib = np.tile(X_calib, (reps, 1))[:n_needed]
    n_img = X_calib.shape[0]
    n_chunks = int(np.ceil(n_img / B))
    Xpad = np.zeros((n_chunks * B, n_in))
    Xpad[:n_img] = X_calib
    rates_vals = Xpad.reshape(n_chunks, B * n_in) * float(p["max_rate"] / Hz)
    hp = dict(r_target=r_target, tau_r=tau_r, eta=eta,
              theta_min=p["v_rest"] + 2 * mV if theta_min is None else theta_min,
              theta_max=p["v_th"] + 60 * mV if theta_max is None else theta_max)
    net = build_network(rates_vals, W, p, homeostasis=hp)
    hid = net["hid"]
    hid.r = 0 * Hz
    trace = []
    for c in range(n_chunks):
        net.run(p["t_enc"])
        trace.append(np.asarray(hid.theta[:n_h] / mV).copy())
    theta_B = np.asarray(hid.theta / volt).reshape(B, n_h)
    theta_out = theta_B.mean(axis=0) * volt      # average over the B replicas
    if verbose:
        print(f"  homeostasis: theta mean={np.mean(theta_out/mV):.1f} mV  "
              f"min={np.min(theta_out/mV):.1f}  max={np.max(theta_out/mV):.1f} mV")
    return theta_out, np.array(trace)


def demo_run(X, W, p, theta_frozen=None, record_neurons=(0, 1, 2)):
    """Small run (a few images, batch=1) returning rasters + voltage traces for plots."""
    defaultclock.dt = p["dt"]
    q = dict(p, batch=1)
    n_img = X.shape[0]
    rates_vals = X.reshape(n_img, p["n_in"]) * float(p["max_rate"] / Hz)
    net = build_network(rates_vals, W, q, theta_frozen=theta_frozen, record_v=list(record_neurons))
    net.run(n_img * p["t_enc"])
    out = dict(
        in_t=np.asarray(net["mon_in"].t / ms), in_i=np.asarray(net["mon_in"].i),
        hid_t=np.asarray(net["mon_hid"].t / ms), hid_i=np.asarray(net["mon_hid"].i),
        v_t=np.asarray(net["mon_v"].t / ms), v=np.asarray(net["mon_v"].v / mV),
        v_neurons=list(record_neurons),
    )
    return out


# ----------------------------------------------------------------------------
# Read-out
# ----------------------------------------------------------------------------
def train_readout(F_train, y_train, F_test, y_test, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=3000))
    clf.fit(F_train, y_train)
    pred = clf.predict(F_test)
    return dict(acc=accuracy_score(y_test, pred),
                f1=f1_score(y_test, pred, average="macro"),
                train_acc=clf.score(F_train, y_train),
                clf=clf, pred=pred)


def accuracy_vs_window(counts_tr, y_tr, counts_te, y_te, p):
    """Accuracy when the read-out only sees spikes from the first k time bins."""
    n_bins = counts_tr.shape[2]
    bin_ms = float(p["t_enc"] / ms) / n_bins
    windows, accs = [], []
    for k in range(1, n_bins + 1):
        Ftr = counts_tr[:, :, :k].sum(axis=2)
        Fte = counts_te[:, :, :k].sum(axis=2)
        r = train_readout(Ftr, y_tr, Fte, y_te)
        windows.append(k * bin_ms)
        accs.append(r["acc"])
    return np.array(windows), np.array(accs)
