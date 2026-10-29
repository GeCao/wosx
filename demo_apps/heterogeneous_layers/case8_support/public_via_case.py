"""Deterministically derive a documented three-net subproblem from case8.cap3d.

This is not a parser for all CAP3D semantics and not a reproduction of the
published DSPF. Explicit nonplanar media, other conductors, and the original
window are omitted. Original selected block dimensions and plate values are
retained. One input length unit is assigned 1 micrometre for this benchmark.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/"demo_apps/heterogeneous_layers/case8/case8.cap3d"


def field(text, name):
    match = re.search(r"^\s*"+name+r"\s+([^\n]+)", text, re.M)
    if not match:
        raise ValueError(f"Missing {name}")
    return match.group(1).strip()


def parse(source=SOURCE):
    text = source.read_text()
    layers = {int(field(v, "id")): {"name": field(v, "name"), "type": field(v, "type")}
              for v in re.findall(r"<layer>(.*?)</layer>", text, re.S)}
    conductors = []
    for net in re.findall(r"<conductor>(.*?)</conductor>", text, re.S):
        blocks = []
        for block in re.findall(r"<block>(.*?)</block>", net, re.S):
            vectors = []
            for name in ("basepoint", "v1", "v2", "hvector"):
                match = re.search(name+r"\(([^)]+)\)", block)
                vectors.append(np.array([float(s) for s in match.group(1).split(",")]))
            base, a, b, h = vectors
            if np.any(np.array([a, b, h])-np.diag(np.diag([a, b, h]))):
                raise ValueError("Non-axis-aligned public block")
            layer = int(field(block, "layer"))
            blocks.append(dict(layer=layer, **layers[layer],
                               block_name=field(block, "name"),
                               bounds=np.round([base, base+a+b+h], 8).tolist()))
        conductors.append(dict(id=field(net, "name"), blocks=blocks))
    plates = []
    for plate in re.findall(r"<plate_medium>(.*?)</plate_medium>", text, re.S):
        plates.append(dict(id=field(plate, "name"), epsilon_r=float(field(plate, "diel")),
                           z_top=float(field(plate, "z_top"))))
    return conductors, plates


def make_case():
    nets, plates = parse()
    selections = {"1": {18, 19, 21}, "2": {18, 19, 21}, "4": {18, 24, 25}}
    # Documented artificial cutout enclosure, in original input coordinates.
    outer = np.array([[18.2, -12.38, .04], [18.68, -11.86, .43]])
    origin, multiplier = np.array([18.44, -12.12, .20]), 10.
    transform = lambda x: np.round((np.array(x)-origin)*multiplier, 10).tolist()
    chosen, selection_log = [], []
    for net in nets:
        if net["id"] not in selections:
            continue
        blocks = [b for b in net["blocks"] if b["layer"] in selections[net["id"]]]
        chosen.append(dict(id="net_"+net["id"], public_net=net["id"],
                           boxes=[transform(b["bounds"]) for b in blocks]))
        selection_log.append(dict(public_net=net["id"], blocks=blocks,
                                  omitted_blocks=len(net["blocks"])-len(blocks)))
    slabs = []
    previous = -.1
    for p in plates:
        low, high = max(previous, outer[0, 2]), min(p["z_top"], outer[1, 2])
        previous = p["z_top"]
        if low >= high:
            continue
        if slabs and slabs[-1]["epsilon_r"] == p["epsilon_r"]:
            slabs[-1]["bounds"][1][2] = high
            slabs[-1]["public_plate_names"].append(p["id"])
        else:
            b = outer.copy(); b[0, 2], b[1, 2] = low, high
            slabs.append(dict(bounds=b.tolist(), epsilon_r=p["epsilon_r"], public_plate_names=[p["id"]]))
    regions = [dict(id=f"slab_{i}", bounds=transform(s["bounds"]),
                    epsilon_r=s["epsilon_r"], public_plate_names=s["public_plate_names"])
               for i, s in enumerate(slabs[1:], 1)]
    source_record = SOURCE.parent/"SOURCE_REVISION.json"
    if source_record.exists():
        identity = json.loads(source_record.read_text())
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != identity["case8_sha256"]:
            raise ValueError("Distributed public source identity changed")
        commit = identity["commit"]
    else:
        commit = subprocess.check_output(["git", "-C", str(SOURCE.parents[2]), "rev-parse", "HEAD"], text=True).strip()
    case = dict(schema_version=5, name="deeprwcap_case8_via_cutout", problem_kind="capacitance",
                reference="grounded_box", length_unit_m=1e-7, outer_box=transform(outer),
                background_epsilon_r=slabs[0]["epsilon_r"], dielectric_regions=regions,
                conductors=chosen, core_padding=[.027]*3, support_padding=[.25]*3,
                provenance=dict(repository="https://github.com/THU-numbda/deepRWCap",
                    commit=commit, path="testcases/cap3d/case8.cap3d",
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    classification="public-derived reduced benchmark; not full-deck reproduction",
                    source_coordinate_origin=origin.tolist(), multiplier=multiplier,
                    assigned_source_length_unit_m=1e-6,
                    length_unit_note="Physical unit assigned for this reduced benchmark; CAP3D file has no explicit unit declaration.",
                    replacement_window_source_coordinates=outer.tolist(),
                    selected_nets=selection_log, retained_merged_slabs=slabs,
                    omitted="All unselected blocks/nets, explicit medium coatings, substrate pieces, and original window.",
                    plate_convention="Each z_top closes the layer extending from the previous z_top; adjacent equal epsilon layers merged.",
                    reference_note="All six replacement-window faces grounded. Public DSPF values are not comparable."))
    return case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/case8/reduced-case.json")
    a = p.parse_args()
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    c = make_case()
    out.write_text(json.dumps(c, indent=2)+"\n")
    print(json.dumps(dict(output=str(out), terminals=len(c["conductors"]),
        blocks=[len(t["boxes"]) for t in c["conductors"]],
        regions=len(c["dielectric_regions"])+1,
        eps=[c["background_epsilon_r"]]+[r["epsilon_r"] for r in c["dielectric_regions"]],
        source_sha256=c["provenance"]["source_sha256"]), indent=2))


if __name__ == "__main__":
    main()
