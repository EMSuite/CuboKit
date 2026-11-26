"""CuboBragg disk matrix: py4DSTEM's find_Bragg_disks on nanobeam (NBED) patterns of many crystals, zones, convergence
angles and doses, simulated with abTEM, so the finder is checked on discs like a user's, not only on planted ones (the
reference matrix, suite plan of 26 Nov 2026; EMS#1153 found an Index finder that missed broad NBED discs).

Writes Tests/CuboBraggTests/Fixtures/bragg-matrix.json and bragg-matrix.f64 in the layout of tools/bragg/pins.py
(scenes of patterns with a kernel, cases with py4DSTEM's keywords and its peaks), plus per case:

* the vacuum probe, py4DSTEM's probe radius (`get_probe_size`) and its sigmoid kernel (`Probe.get_kernel(mode="sigmoid",
  radii=(r, 2 r))`, the tutorial's), so the kit's kernel is checked too;
* the truth: every zero-order reflection of the crystal on the detector (its pixel from the box's reciprocal lattice and
  the wavelength), and which of them are strong (the disc holds at least 2 % of the strongest diffracted disc, inside the
  edge boundary); `recall` is the share of strong discs py4DSTEM finds inside the disc (within the probe radius, at
  least 1.5 px). In a 20 nm crystal a disc is brightest near its exact Bragg condition, its inner edge, so the
  correlation maximum can sit 10 % of g inside the disc's geometric centre: that is the physics, not a miss.

Patterns: abTEM 1.0.10, float64, Lobato, infinite projection, 1 A slices, sampling 0.1 A, the crystal repeated to at least
60 A across and about 20 nm deep, a probe at the box centre (no aberrations), 200 kV. The pattern is read onto a
128 x 128 detector of 0.625 mrad per pixel (bilinear, scipy's map_coordinates order 1), the direct beam at (64, 64).
Convergence 2 and 4 mrad (disc radii about 3 and 6 px); doses: noiseless and 2e4 electrons (Poisson, seeded).

Run from the repo root, in the tools/requirements.txt environment plus tools/bragg-matrix/requirements.txt:
    .venv/bin/python -I -B tools/bragg-matrix/pins.py
"""
from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import map_coordinates
import abtem
import ase
from ase.build import bulk, surface
from ase.spacegroup import crystal
import py4DSTEM
from py4DSTEM.braggvectors.diskdetection import find_Bragg_disks
from py4DSTEM.process.calibration import get_probe_size

assert abtem.__version__ == "1.0.10" and ase.__version__ == "3.29.0"
assert py4DSTEM.__version__ == "0.14.19" and np.__version__ == "2.5.4" and scipy.__version__ == "1.18.1"
abtem.config.set({"precision": "float64", "device": "cpu", "fft": "numpy"})

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Tests" / "CuboBraggTests" / "Fixtures"
ENERGY = 200e3
WAVELENGTH = abtem.core.energy.energy2wavelength(ENERGY)  # A
DETECTOR = 128
MRAD_PER_PX = 0.625
LATERAL = 60.0
DEPTH = 200.0
ALPHAS = (2.0, 4.0)
DOSES = (None, 2e4)
SEED = 20261126


def structures():
    s = {}
    s["Al"] = bulk("Al", "fcc", a=4.05, cubic=True)
    s["Au"] = bulk("Au", "fcc", a=4.078, cubic=True)
    s["Si"] = bulk("Si", "diamond", a=5.431, cubic=True)
    s["MgO"] = bulk("MgO", "rocksalt", a=4.212, cubic=True)
    s["Fe"] = bulk("Fe", "bcc", a=2.8665, cubic=True)
    s["SrTiO3"] = crystal(["Sr", "Ti", "O"], [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)], spacegroup=221,
                          cellpar=[3.905, 3.905, 3.905, 90, 90, 90])
    s["Mg"] = crystal("Mg", [(1 / 3, 2 / 3, 0.25)], spacegroup=194, cellpar=[3.209, 3.209, 5.211, 90, 90, 120])
    s["WS2"] = crystal(["W", "S"], [(1 / 3, 2 / 3, 0.25), (1 / 3, 2 / 3, 0.621)], spacegroup=194,
                       cellpar=[3.153, 3.153, 12.323, 90, 90, 120])
    s["TiO2"] = crystal(["Ti", "O"], [(0, 0, 0), (0.3053, 0.3053, 0)], spacegroup=136,
                        cellpar=[4.594, 4.594, 2.959, 90, 90, 90])
    return s


# Zones whose (hkl) plane normal is the zone axis (cubic any; hexagonal and tetragonal [001]).
ZONES = [("Al", (0, 0, 1)), ("Al", (1, 1, 0)), ("Au", (1, 1, 1)), ("Si", (1, 1, 0)), ("Si", (0, 0, 1)),
         ("MgO", (0, 0, 1)), ("Fe", (0, 0, 1)), ("Fe", (1, 1, 1)), ("SrTiO3", (0, 0, 1)), ("Mg", (0, 0, 1)),
         ("WS2", (0, 0, 1)), ("TiO2", (0, 0, 1))]


def unit(atoms, zone):
    slab = surface(atoms, zone, layers=1, periodic=True)
    return abtem.orthogonalize_cell(slab)


def reflections(cell):
    """The zero-order reflections on the detector: (n, m) of the orthogonal cell's projected lattice with a projected
    structure factor (atomic numbers as weights) above 1e-3 of F(0)."""
    ax, ay = cell.cell.lengths()[:2]
    frac = cell.get_scaled_positions()
    z = cell.numbers.astype(float)
    gmax = (DETECTOR / 2) * MRAD_PER_PX * 1e-3 / WAVELENGTH
    out = []
    for n in range(-int(gmax * ax) - 1, int(gmax * ax) + 2):
        for m in range(-int(gmax * ay) - 1, int(gmax * ay) + 2):
            gx, gy = n / ax, m / ay
            if math.hypot(gx, gy) > gmax:
                continue
            f = abs(np.sum(z * np.exp(2j * np.pi * (n * frac[:, 0] + m * frac[:, 1]))))
            if f > 1e-3 * z.sum():
                px = DETECTOR / 2 + gx * WAVELENGTH * 1e3 / MRAD_PER_PX
                py = DETECTOR / 2 + gy * WAVELENGTH * 1e3 / MRAD_PER_PX
                out.append((n, m, px, py))
    return out


def detector(pattern):
    """A DiffractionPatterns' intensity read onto the 128 x 128 detector, the direct beam at (64, 64)."""
    arr = np.asarray(pattern.array)  # abTEM shifts the zero frequency to n // 2 already
    nx, ny = arr.shape[-2:]
    sx, sy = pattern.angular_sampling  # mrad per pixel along x and y
    q = (np.arange(DETECTOR) - DETECTOR / 2) * MRAD_PER_PX
    cx, cy = np.meshgrid(q / sx + nx // 2, q / sy + ny // 2, indexing="ij")
    return map_coordinates(arr, [cx, cy], order=1, mode="constant", cval=0.0)


def simulate(cell, alpha):
    ax, ay = cell.cell.lengths()[:2]
    reps = (max(1, math.ceil(LATERAL / ax)), max(1, math.ceil(LATERAL / ay)),
            max(1, round(DEPTH / cell.cell.lengths()[2])))
    atoms = cell * reps
    pot = abtem.Potential(atoms, sampling=0.1, slice_thickness=1.0, projection="infinite", parametrization="lobato")
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=alpha)
    probe.grid.match(pot)
    centre = [tuple(np.array(pot.extent) / 2)]
    exit_wave = probe.multislice(pot, scan=centre).compute()
    dp = detector(exit_wave.diffraction_patterns(max_angle=None).squeeze())
    return dp, reps, atoms.cell.lengths()[2]


def vacuum(alpha):
    """The vacuum probe of a convergence angle on a 60 A, 0.1 A grid, read onto the detector: recorded once, as a user
    records it, and used for every crystal."""
    probe = abtem.Probe(energy=ENERGY, semiangle_cutoff=alpha, extent=LATERAL, sampling=0.1)
    return detector(probe.build(scan=[(LATERAL / 2, LATERAL / 2)]).compute().diffraction_patterns(max_angle=None).squeeze())


arrays, offsets, blob = {}, {}, []


def put(name, a):
    a = np.ascontiguousarray(a, dtype="<f8")
    offsets[name] = {"offset": sum(len(b) for b in blob), "shape": list(a.shape)}
    blob.append(a.ravel())


def main():
    rng = np.random.default_rng(SEED)
    radius, kernels = {}, {}
    s = structures()
    scenes, cases = {}, []
    for name, zone in ZONES:
        cell = unit(s[name], zone)
        refl = reflections(cell)
        for alpha in ALPHAS:
            dp0, reps, depth = simulate(cell, alpha)
            # The vacuum probe depends on the convergence alone: one vacuum and one kernel per angle, shared by its scenes.
            ka = f"k_a{alpha:g}"
            if ka not in offsets:
                vac = vacuum(alpha)
                r = float(get_probe_size(vac)[0])
                probe = py4DSTEM.Probe(data=vac)
                kernel = np.asarray(probe.get_kernel(mode="sigmoid", origin=(DETECTOR / 2, DETECTOR / 2),
                                                     radii=(r, 2 * r)))
                put(ka, kernel)
                put(f"v_a{alpha:g}", vac)
                radius[alpha], kernels[alpha] = r, kernel
            r, kernel = radius[alpha], kernels[alpha]
            # the truth: disc sums around each reflection on the noiseless pattern
            qy, qx = np.mgrid[0:DETECTOR, 0:DETECTOR]
            edge = int(math.ceil(r)) + 2
            sums = []
            for n, m, px, py in refl:
                inside = (qy - px) ** 2 + (qx - py) ** 2 <= r * r  # axis 0 is qx in py4DSTEM's naming
                sums.append(float(dp0[inside].sum()))
            diffracted = [v for (n, m, _, _), v in zip(refl, sums) if (n, m) != (0, 0)]
            top = max(diffracted) if diffracted else 0.0
            strong = [i for i, ((n, m, px, py), v) in enumerate(zip(refl, sums))
                      if (n, m) != (0, 0) and v >= 0.02 * top and edge <= px < DETECTOR - edge and edge <= py < DETECTOR - edge]
            g_px = min((math.hypot(px - DETECTOR / 2, py - DETECTOR / 2) for n, m, px, py in refl if (n, m) != (0, 0)),
                       default=DETECTOR / 4)
            sid = f"{name}{''.join(map(str, zone))}_a{alpha:g}"
            patterns = []
            for dose in DOSES:
                if dose is None:
                    patterns.append(dp0)
                else:
                    patterns.append(rng.poisson(dp0 / dp0.sum() * dose).astype(float))
            put("p_" + sid, np.array(patterns))
            scenes[sid] = {"rows": DETECTOR, "columns": DETECTOR, "kernel": ka, "patterns": "p_" + sid,
                           "vacuum": f"v_a{alpha:g}", "probe_radius": r, "structure": name, "zone": list(zone),
                           "alpha_mrad": alpha, "repeats": list(reps), "thickness": depth,
                           "reflections": [[n, m, px, py] for n, m, px, py in refl], "strong": strong}
            for p, dose in enumerate(DOSES):
                for sub in ("poly", "multicorr"):
                    opt = dict(corrPower=1.0, sigma_dp=0.0, sigma_cc=2.0, subpixel=sub, upsample_factor=16,
                               minAbsoluteIntensity=0.0, minRelativeIntensity=0.005, relativeToPeak=0,
                               minPeakSpacing=float(max(r, 0.5 * g_px)), edgeBoundary=edge, maxNumPeaks=100)
                    res = find_Bragg_disks(patterns[p], kernel, **opt)
                    out = np.column_stack([res.data["qx"], res.data["qy"], res.data["intensity"]])
                    found = 0
                    for i in strong:
                        _, _, px, py = refl[i]
                        if len(out) and np.min(np.hypot(out[:, 0] - px, out[:, 1] - py)) <= max(1.5, r):
                            found += 1
                    recall = found / len(strong) if strong else 1.0
                    cid = f"{sid}_{'clean' if dose is None else 'dose'}_{sub}"
                    put("e_" + cid, out.reshape(-1, 3))
                    jopt = dict(opt)
                    jopt["subpixel"] = ["pixel", "poly", "multicorr"].index(sub)
                    jopt["relativeReferenceMinimumRadiusPx"] = 0
                    cases.append({"id": cid, "scene": sid, "pattern": p, "reference": "py4dstem", "options": jopt,
                                  "expected": "e_" + cid, "recall": recall, "fast": p == 0 and sub == "poly"})
                    print(f"{cid}: r {r:.2f} px, {len(out)} peaks, {len(strong)} strong, recall {recall:.2f}")
    meta = {"generator": "tools/bragg-matrix/pins.py",
            "versions": {"py4DSTEM": py4DSTEM.__version__, "abtem": abtem.__version__, "numpy": np.__version__,
                         "scipy": scipy.__version__, "python": platform.python_version()},
            "energy": ENERGY, "mrad_per_px": MRAD_PER_PX, "arrays": offsets, "scenes": scenes, "cases": cases}
    (OUT / "bragg-matrix.f64").write_bytes(np.concatenate(blob).astype("<f8").tobytes())
    (OUT / "bragg-matrix.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
