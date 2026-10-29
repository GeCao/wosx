# Heterogeneous WoS demos and Case-8 FEM validation

Run these commands from the repository root in the `CAEMLWoS` environment. The
tracked [case-8 deck](case8/case8.cap3d) drives the two FEM demos and two
rectilinear WoS demos. A separate mesh demo accepts PLY or OBJ triangle surfaces.
The WoS demos never run an FEM solve.

```bash
micromamba activate CAEMLWoS
cmake -S . -B build -DWOSX_ENABLE_GPU_SUPPORT=ON
cmake --build build --target heterogeneous_rectilinear_walk_gpu --parallel 2
```

## FEM 1: convergence

```bash
python demo_apps/heterogeneous_layers/case8_fem_convergence.py --enriched --threads 10
```

This solves the eight-terminal matrix on both meshes at polynomial order 2,
then on the refined mesh with local order 4, far-field order 3, and graded-cell
order capped at 3. Matching results are reused; `--recompute` forces new
solves. Outputs are in `data/case8/fem/`: `base-order2.json`,
`refined-order2.json`, `refined-order4.json`, `convergence.json`, and
`enrichment.json`. The largest order-2 mesh change was 0.000416 fF; the largest
order-2 to order-4 change was 0.013559 fF. Mesh convergence alone was not an
adequate reference check. The final 19,318,247-degree-of-freedom order-4 solve
had maximum relative residual 1.31e-9.

The local meshes are `assets/case8/fem-focus-hp2-factor02.vol` and
`assets/case8/fem-focus-hp3-factor02.vol`. They were copied from the original
validation checkout. A verified public download for these generated meshes has
not been identified.

## FEM 2: problem and result visualization

```bash
python demo_apps/heterogeneous_layers/case8_fem_visualization.py
```

This saves `data/case8/fem/overview.png`: the geometry, terminal-1 potential
slice, and quadratic FEM matrix heatmap. It uses the saved potential field when
available, or solves terminal 1 to create it.

## WoS 1: geometry, GPU matrix, and FEM comparison

```bash
python demo_apps/heterogeneous_layers/case8_wos_comparison.py --reuse-gpu
```

The command reads the saved order-4 FEM matrix in
`data/case8/fem/refined-order4.json`; it never solves FEM. It verifies the
pinned deck hash, runs all eight GPU charge rows and samples the WoSX potential
on a slice when needed, and saves under `data/case8/wos/`:

- `gpu-wos.json`: raw, unsymmetrized capacitance rows and sampling errors;
- `gpu-paths.npz`: per-path labels and weights for hit intervals;
- `wos-potential.npz`: terminal-1 GPU WoS potential estimates and sampling errors;
- `assessment.json`: FEM difference, precision, and reciprocity checks;
- `comparison.png`: WoSX terminal-1 potential and conductor outlines, plus GPU,
  order-4 FEM, and signed-difference matrix heatmaps. The difference panel
  enlarges its color scale because its values are much smaller.

`--reuse-gpu` reuses saved charge paths and potential samples only when the case,
shader/executable hash, seed, path count, and walk settings match. Without it,
the GPU runs are fresh. Use `--paths`, `--seed`, `--tolerance`, or `--output-dir`
for a new campaign. The default 80 × 64 slice has 3,845 dielectric pixels;
128 paths per pixel required 492,160 potential walks and took about five minutes
on the local Vulkan backend. The largest plug-in pointwise sampling standard
error was 0.044 V. Adjust `--potential-grid NX NZ` and `--potential-paths N`
to trade time for detail and precision.
The default is 100,000 paths per row at tolerance 5e-5; this took about 199
seconds summed over rows on the local Vulkan backend. All 800,000 paths
finished. The largest self-entry sampling error was 0.009064 fF; the largest
reciprocity gap was 0.012055 fF, or 1.79 combined standard errors. Against the
fresh order-4 FEM matrix, the largest difference was 0.019247 fF (2.63
sampling standard errors), and no entry fell outside the conservative 95%
familywise hit intervals. These checks support agreement at the measured
precision; they do not certify zero stopping-tolerance or FEM discretization
error.

## WoS 2: paired tolerance convergence

```bash
python demo_apps/heterogeneous_layers/case8_wos_convergence.py \
  --paths 2000 --fem-reference data/case8/fem/refined-order4.json
```

This uses the same signed charge starts at tolerances 2e-4, 1e-4, 5e-5, and
2.5e-5, and records paired terminal changes in
`data/case8/wos/convergence/`. It is an optional convergence diagnostic.
The saved pilot campaigns were removed; this command regenerates them.

## Triangle-mesh WoS: Spot or another PLY/OBJ conductor

```bash
cmake --build build --target heterogeneous_triangle_mesh_walk_gpu --parallel 2
python demo_apps/heterogeneous_layers/triangle_mesh_potential_demo.py assets/spot.ply --reuse
```

Pass any PLY or OBJ triangle mesh as the positional argument. The surface is a
1 V absorbing conductor inside a grounded rectangular box. A horizontal plane
splits the dielectric into two constant relative permittivities (default 2
below, 8 above). The GPU walker uses WoSX's triangle BVH to find the mesh and
the chapter-5 permittivity-weighted hemisphere choice at the dielectric plane.
The sampling slice sits in front of the mesh, so all its starts are in the
dielectric. The 3D mesh and slice are shown separately in the figure.

The default run samples 64 × 64 pixels with 48 paths each (196,608 walks) and
saves `data/triangle_mesh/spot/potential.png`, `potential.npz`, and
`report.json`. Spot has 3,225 vertices and 5,856 triangles; all default walks
reached a boundary. `--reuse` uses a saved field only when the mesh, compiled
runner, shader, and settings match. Use `--grid NX NZ`, `--paths N`,
`--eps-low`, `--eps-high`, `--interface-z`, `--padding`, and `--output-dir` to
change the problem. No volume mesh or FEM reference is needed.

This mesh path estimates **potential for one conducting triangle surface**.
It does not yet compute a capacitance matrix or accept curved dielectric
interfaces, several independently biased conductors, or a volumetric material
field. For those, the rectilinear Case-8 path remains the validated solver.

## Physical problem and provenance

One coordinate unit is 0.1 µm. The window is
`[-37.626, -38.766, -3.0]` to `[46.126, 38.126, 94.558]` in those units.
The deck has nine conductors (127 metal boxes, including `GROUND`), 57
background dielectric plates, and 150 ordered coating blocks. The relative
permittivity is piecewise constant, from 1 through 20. Metal overrides
material, and later coatings override earlier ones.

Both methods solve the charge-free equation `∇·(ε₀ εᵣ ∇φ) = 0` in the
dielectric. For each active terminal, its voltage is 1 V and all other
conductors are at 0 V. The outer window has zero normal flux. The result is
the 8×8 Maxwell capacitance matrix in fF with `GROUND` as reference. The FEM
solver uses NGSolve; the WoS solver samples the signed weak charge measure on
the CPU and traces terminal hits in WoSX's Slang/RHI GPU kernel. No Zombie
source or archived capacitance matrix is required.

The deck's MIT license and pinned source revision are beside the input file.
The importer and optional FEM solver provenance is described in
[`case8_support/README.md`](case8_support/README.md).
