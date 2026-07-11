#!/usr/bin/env python3
"""Combine the three pre-coffee experiment outputs into one coffee brief."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from othello_common import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="out/precoffee")
    args = parser.parse_args()
    out = Path(args.out_dir)
    legal = json.loads((out / "legal" / "legal_eval.json").read_text())
    steering = json.loads((out / "steering" / "steering_results.json").read_text())
    workspace = json.loads((out / "workspace" / "workspace_split.json").read_text())
    layers = legal["source_layers"]
    final_legal = legal["metrics"]["final"]
    strongest_alpha = max(steering["alphas"])
    layer6 = layers[-1]
    j_steer = steering["summary"][f"legal|jlens|L{layer6}|a{strongest_alpha:g}"][
        "delta_target_logit"
    ]["mean"]
    random_steer = steering["summary"][f"legal|random|L{layer6}|a{strongest_alpha:g}"][
        "delta_target_logit"
    ]["mean"]
    j_variance = workspace["geometry"][str(layer6)]["activation_variance_fraction"]
    remove_j = workspace["ablations"][f"remove_jspace|L{layer6}"]["legal_probability_mass"]["mean"]
    remove_random = workspace["ablations"][f"remove_random|L{layer6}"]["legal_probability_mass"][
        "mean"
    ]
    baseline_mass = workspace["ablations"]["baseline"]["legal_probability_mass"]["mean"]
    combined = {
        "legal": legal,
        "steering": steering,
        "workspace": workspace,
    }
    write_json(out / "results.json", combined)
    lines = [
        "# OthelloGPT pre-coffee workspace brief",
        "",
        "## Read: legal-action content",
        "",
        f"Final legal precision@5: "
        f"{final_legal['legal_precision_at_k']['mean']:.3f}; final legal probability "
        f"mass: {final_legal['legal_probability_mass']['mean']:.3f}.",
        "",
        "![Legal precision](legal/legal_precision_by_layer.svg)",
        "",
        "## Write: causal move steering",
        "",
        f"At L{layer6} and alpha={strongest_alpha:g}, mean legal-target logit change "
        f"is {j_steer:.3f} for J-steering versus {random_steer:.3f} for random control.",
        "",
        "![Steering](steering/target_logit_effect_by_layer.svg)",
        "",
        "## Selectivity: small subspace, disproportionate function",
        "",
        f"At L{layer6}, the J-span contains {j_variance:.1%} of activation energy. "
        f"Baseline legal mass is {baseline_mass:.3f}; after J-space removal it is "
        f"{remove_j:.3f}, versus {remove_random:.3f} after matched random removal.",
        "",
        "![Workspace ablation](workspace/ablation_effect_by_layer.svg)",
        "",
        "## Interpretation rule",
        "",
        "Evidence for a tiny-model workspace requires all three: readable legal-action "
        "content, target-specific causal writing, and selective functional importance "
        "relative to matched controls. Any missing leg weakens the workspace account.",
        "",
    ]
    (out / "precoffee_summary.md").write_text("\n".join(lines))
    print(f"saved combined brief to {out / 'precoffee_summary.md'}")


if __name__ == "__main__":
    main()
