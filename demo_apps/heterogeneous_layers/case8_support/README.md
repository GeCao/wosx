# Case-8 importer provenance

The tracked MIT-licensed public deck is in
`demo_apps/heterogeneous_layers/case8/case8.cap3d`, with its pinned source
revision and license alongside it. `full_case8.py` and `public_via_case.py`
were adapted from `CAEML_original/CAEML-WoS/research/scripts/`; they regenerate
the rectilinear case directly from that deck. The reusable coordinate arrangement and signed charge measure are implemented
in `heterogeneous_wos/rectilinear_case.py`. The WoSX GPU binary adapter is in
`heterogeneous_wos/gpu_rectilinear.py`. The optional native Zombie scene has
been removed.

`fem_fullcase_iterative.py` was adapted from the same source release. It is
shared by the two FEM demos and computes a fresh independent NGSolve matrix
on the local meshes in `assets/case8/`. The mesh came from
`caeml-validation-data/research/results/fullcase-v0.6/case8/`. NGSolve is an
optional external dependency; it is not bundled in this repository.

Copied Zombie source/build trees and archived capacitance results have been
removed. The GPU case-8 demo needs only the tracked deck, the importer, and
WoSX's own GPU stack.
