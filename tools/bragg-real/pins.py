"""CuboBragg on real public data: py4DSTEM's find_Bragg_disks, run as its tutorials run it, on two recorded 4D-STEM
datasets from Zenodo (both CC-BY-4.0), so the finder is held on detector noise, gamma and chip gaps, not only on
simulated discs (the reference matrix, suite plan of 26 Nov 2026; tools/bragg-matrix is the simulated half).

* `nanowire`: pyxem's demo data, Zenodo 14113591 (Carter Francis, 2024), `twinned_nanowire.hdf5`: a precession (2 deg)
  scan of a twinned GaAs nanowire, 100 x 30 positions, 144 x 144 uint8 patterns at 300 kV.
* `nio`: Andy Bridger, "Mixed Phase Test Datasets for py4dstem", Zenodo 8233585 (2023), `20210306_114758.hdf5`: NiO and layered oxide
  grains, 254 x 255 positions, 384 x 384 uint8 patterns on a four-chip detector (its gaps are in the data).

The recipe is py4DSTEM's (tutorial orient_strain_01_WS2, whose datasets have no vacuum probe either):
  1. the mean pattern of the whole scan (`DataCube.get_dp_mean`), its probe radius (`get_probe_size`);
  2. a synthetic probe of that radius (`Probe.generate_synthetic_probe(radius, width=0.7, Qshape)`) and its sigmoid
     kernel (`get_kernel(mode='sigmoid', origin=(Q/2, Q/2), radii=(1.2 r, 4 r))`);
  3. `find_Bragg_disks(template=kernel, corrPower=1, sigma=0, edgeBoundary=4, minRelativeIntensity=0,
     subpixel='poly', upsample_factor=8, maxNumPeaks=1000)` (the tutorial's keywords), with the two the tutorial says to
     tune per dataset: `minAbsoluteIntensity` just above the strongest noise peak of the off-sample positions (nanowire
     8, noise peaks reach 6.7; nio 1, noise peaks reach 0.79) and `minPeakSpacing` under the spot spacing (5 and 8 px).

Patterns: nanowire scan positions NANOWIRE (along the wire, and one off it); nio the 6 positions of the 8-step grid with
the most peaks at these settings, and one off-sample position. They are stored in their recorded type (uint8).

Writes Tests/CuboBraggTests/Fixtures/bragg-real.json (scenes, cases, py4DSTEM's peaks), bragg-real.f64 (mean patterns,
kernels, peaks) and bragg-real.u8 (the patterns). The datasets are downloaded once to $BRAGG_REAL_DATA (default
../References/CuboKit/zenodo) and checked against Zenodo's MD5.

    ../References/python/py4dstem/bin/python -I -B tools/bragg-real/pins.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import urllib.request
import warnings
from pathlib import Path

import h5py
import numpy as np
import scipy

warnings.filterwarnings("ignore")
import py4DSTEM  # noqa: E402
from py4DSTEM.braggvectors.diskdetection import find_Bragg_disks  # noqa: E402
from py4DSTEM.process.calibration import get_probe_size  # noqa: E402

assert py4DSTEM.__version__ == "0.14.19"

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Tests" / "CuboBraggTests" / "Fixtures"
DATA = Path(os.environ.get("BRAGG_REAL_DATA", ROOT.parent / "References" / "CuboKit" / "zenodo"))

SOURCES = {
    "nanowire": {"record": 14113591, "file": "twinned_nanowire.hdf5", "md5": "2765fa855db8bc011252dbb425facc81",
                 "minAbsoluteIntensity": 8.0, "minPeakSpacing": 5.0},
    "nio": {"record": 8233585, "file": "20210306_114758.hdf5", "md5": "7565d905fb2cb413aef9be8e6207dfe7",
            "minAbsoluteIntensity": 1.0, "minPeakSpacing": 8.0},
}
NANOWIRE = [(20, 15), (40, 15), (60, 15), (80, 15), (8, 8), (98, 22), (0, 27)]
NIO_GRID, NIO_PICK, NIO_OFF = 8, 6, (0, 0)
TUTORIAL = dict(corrPower=1.0, sigma_dp=0.0, sigma_cc=0.0, edgeBoundary=4, minRelativeIntensity=0.0, relativeToPeak=0,
                subpixel="poly", upsample_factor=8, maxNumPeaks=1000)


def fetch(src):
    path = DATA / src["file"]
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        url = f"https://zenodo.org/records/{src['record']}/files/{src['file']}?download=1"
        urllib.request.urlretrieve(url, path)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    assert digest == src["md5"], f"{path}: md5 {digest}, Zenodo says {src['md5']}"
    return path


def mean_pattern(d):
    """DataCube.get_dp_mean: np.mean over the scan, streamed row by row (the nio cube is 9.5 GB as uint8)."""
    acc = np.zeros(d.shape[2:], np.float64)
    for i in range(d.shape[0]):
        acc += d[i].astype(np.float64).sum(axis=0)
    return acc / (d.shape[0] * d.shape[1])


def kernel_for(dp_mean):
    r = float(get_probe_size(dp_mean)[0])
    q = dp_mean.shape
    probe = py4DSTEM.Probe.generate_synthetic_probe(radius=r, width=0.7, Qshape=q)
    kernel = np.asarray(probe.get_kernel(mode="sigmoid", origin=(q[0] / 2, q[0] / 2), radii=(r * 1.2, r * 4)))
    return r, kernel


def detect(pattern, kernel, src):
    opt = dict(TUTORIAL, minAbsoluteIntensity=src["minAbsoluteIntensity"], minPeakSpacing=src["minPeakSpacing"])
    res = find_Bragg_disks(pattern.astype(np.float64), kernel, **opt)
    return opt, np.column_stack([res.data["qx"], res.data["qy"], res.data["intensity"]])


f64, u8, offsets = [], [], {}


def put(name, a, kind="f64"):
    store = f64 if kind == "f64" else u8
    a = np.ascontiguousarray(a, dtype="<f8" if kind == "f64" else np.uint8)
    offsets[name] = {"offset": sum(len(b) for b in store), "shape": list(a.shape), "file": kind}
    store.append(a.ravel())


def main():
    scenes, cases = {}, []
    for sid, src in SOURCES.items():
        with h5py.File(fetch(src), "r") as h:
            d = h["Experiments/__unnamed__/data"]
            dp_mean = mean_pattern(d)
            r, kernel = kernel_for(dp_mean)
            if sid == "nanowire":
                picks = NANOWIRE
            else:
                grid = [(i, j) for i in range(0, d.shape[0], NIO_GRID) for j in range(0, d.shape[1], NIO_GRID)]
                counts = [len(detect(d[i, j], kernel, src)[1]) for i, j in grid]
                order = sorted(range(len(grid)), key=lambda k: (-counts[k], k))[:NIO_PICK]
                picks = [grid[k] for k in order] + [NIO_OFF]
            patterns = np.stack([d[i, j] for i, j in picks])
        rows, columns = dp_mean.shape
        put("m_" + sid, dp_mean)
        put("k_" + sid, kernel)
        put("p_" + sid, patterns, "u8")
        scenes[sid] = {"rows": rows, "columns": columns, "mean": "m_" + sid, "kernel": "k_" + sid, "patterns": "p_" + sid,
                       "probe_radius": r, "kernel_radii": [1.2 * r, 4 * r], "synthetic_width": 0.7,
                       "zenodo": src["record"], "file": src["file"], "md5": src["md5"], "positions": [list(p) for p in picks]}
        for p, pos in enumerate(picks):
            opt, out = detect(patterns[p], kernel, src)
            cid = f"{sid}_{pos[0]}_{pos[1]}"
            put("e_" + cid, out.reshape(-1, 3))
            jopt = dict(opt)
            jopt["subpixel"] = ["pixel", "poly", "multicorr"].index(opt["subpixel"])
            jopt["relativeReferenceMinimumRadiusPx"] = 0
            cases.append({"id": cid, "scene": sid, "pattern": p, "options": jopt, "expected": "e_" + cid})
            print(f"{cid}: r {r:.3f} px, {len(out)} peaks")
    meta = {"generator": "tools/bragg-real/pins.py",
            "versions": {"py4DSTEM": py4DSTEM.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
                         "h5py": h5py.__version__, "python": platform.python_version()},
            "licence": "CC-BY-4.0 (both Zenodo records)", "arrays": offsets, "scenes": scenes, "cases": cases}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bragg-real.f64").write_bytes(np.concatenate(f64).astype("<f8").tobytes())
    (OUT / "bragg-real.u8").write_bytes(np.concatenate(u8).astype(np.uint8).tobytes())
    (OUT / "bragg-real.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
