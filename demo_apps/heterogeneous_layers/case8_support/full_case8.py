"""Lossless rectilinear case-8 importer, with explicit interpretation records.

All conductor and coating blocks, all plate records, and the original window
are retained. Named conductor regions remain distinct; a separate projection
can aggregate the two signalGND regions for comparison with the supplied DSPF.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
from .public_via_case import field, parse, SOURCE


def block_bounds(text):
    values = [np.array([float(x) for x in re.search(n+r"\(([^)]+)\)", text).group(1).split(",")])
              for n in ("basepoint", "v1", "v2", "hvector")]
    base, a, b, h = values
    if np.any(np.array([a, b, h])-np.diag(np.diag([a, b, h]))):
        raise ValueError("Nonrectilinear block")
    bounds = np.round([base, base+a+b+h], 8)
    if np.any(bounds[0] >= bounds[1]):
        raise ValueError("Nonpositive block")
    return bounds


def make_case():
    text = SOURCE.read_text()
    nets, plates = parse()
    window = re.search(r"<window>(.*?)</window>", text, re.S).group(1)
    outer = np.array([[float(x) for x in re.search(n+r"\(([^)]+)\)", window).group(1).split(",")]
                      for n in ("v1", "v2")])
    origin, multiplier = np.array([18.44, -12.12, .2]), 10.
    transform = lambda x: np.round((np.asarray(x)-origin)*multiplier, 10).tolist()
    metals = [dict(id=n["id"], boxes=[transform(b["bounds"]) for b in n["blocks"]],
                   original_blocks=n["blocks"]) for n in nets]
    media = []
    for order, v in enumerate(re.findall(r"<medium>(.*?)</medium>", text, re.S)):
        blocks = re.findall(r"<block>(.*?)</block>", v, re.S)
        for block in blocks:
            media.append(dict(id=field(v, "name"), epsilon_r=float(field(v, "diel")),
                              layer=int(field(block, "layer")), file_order=order,
                              bounds=transform(block_bounds(block))))
    # process_layers() in the published binary assigns increasing priorities
    # to successive medium blocks; higher priorities take precedence. Layer
    # IDs describe process layers and are not the overlap-priority index.
    slab = []
    low = outer[0, 2]
    for p in plates:
        if p["z_top"] <= low:
            raise ValueError("Nonincreasing plate top")
        b = outer.copy(); b[0, 2], b[1, 2] = low, p["z_top"]
        slab.append(dict(id=p["id"], epsilon_r=p["epsilon_r"], bounds=transform(b)))
        low = p["z_top"]
    if low != outer[1, 2]:
        raise ValueError("Plates do not fill the original window")
    return dict(schema_version=6, name="deeprwcap_case8_full", problem_kind="capacitance",
        reference="insulating_window", reference_terminal="GROUND", length_unit_m=1e-7,
        outer_box=transform(outer), conductors=metals, plates=slab, coatings=media,
        provenance=dict(repository="https://github.com/THU-numbda/deepRWCap",
            revision="117c26612467239f912187e21fd0cf594fe29d1e",
            source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            source_path="testcases/cap3d/case8.cap3d", omitted_blocks=0,
            source_origin=origin.tolist(), multiplier=multiplier,
            length_interpretation="One source unit is interpreted as micrometre; retain normalized results separately.",
            outer_boundary_evidence="DeepRWCap AAAI 2026 equation (3): homogeneous Neumann extraction window.",
            material_precedence="Metal overrides dielectric. Plate background is overwritten by explicit coatings in source file order; later blocks take precedence.",
            precedence_evidence="Static audit of public librwcapall.so: process_layers() at 0xe2325--0xe2346 increments and assigns medium-block priority in traversal order; build_relation_for_conformal_medium() sorts these priorities in descending order. No published binary executed on this platform.",
            terminal_mapping="All nine named conductor regions retained; two __signalGND regions aggregate only in a separate DSPF comparison.",
            full_deck_geometry=True, public_dspf_is_converged_reference=False))


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="data/case8/full-case.json")
    a=p.parse_args();case=make_case();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(case,indent=2)+"\n")
    boxes=np.concatenate([c["boxes"] for c in case["conductors"]]);tags=np.concatenate([np.full(len(c["boxes"]),i) for i,c in enumerate(case["conductors"])])
    gap=np.inf;pair=None
    for i,b in enumerate(boxes):
        for j in range(i):
            if tags[i]==tags[j]:continue
            sep=np.maximum(np.maximum(b[0]-boxes[j,1],boxes[j,0]-b[1]),0.)
            d=np.linalg.norm(sep)
            if d<gap:gap=d;pair=[int(tags[i]),int(tags[j])]
    summary=dict(terminals=[c["id"] for c in case["conductors"]],metal_blocks=len(boxes),
                 coatings=len(case["coatings"]),plates=len(case["plates"]),minimum_terminal_gap=gap,
                 minimum_gap_terminals=pair,outer_box=case["outer_box"],source_sha256=case["provenance"]["source_sha256"])
    out.with_name("import-audit.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
