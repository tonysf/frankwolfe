"""Declare the calibration grid before execution; no scientific computation."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--commit", required=True)
args = parser.parse_args()

trials = []
for channel in (0, 1):
    for alpha in (5e-6, 5e-5, 5e-4, 5e-3):
        for multiplier in (.5, 1., 2., 4., 8., 16.):
            if multiplier == 1 and alpha in (5e-6, 5e-4):
                continue  # These GM runs already exist, including their 10k checkpoints.
            trials.append(dict(id=f"c{channel}-printed-a{alpha:g}-s{multiplier:g}".replace(".", "p"),
                               method="printed-adafgd", channel=channel,
                               initialization_seed=200000 + 10*channel,
                               alpha=alpha, eta_scale=multiplier,
                               n_steps=10000, metrics_frequency=100))
    for alpha in (5e-5, 5e-4, 5e-3):
        trials.append(dict(id=f"c{channel}-exact-ray-a{alpha:g}".replace(".", "p"),
                           method="exact-linesearch-fgd", channel=channel,
                           initialization_seed=200000 + 10*channel,
                           alpha=alpha, eta_scale=1.,
                           n_steps=10000, metrics_frequency=100))

plan = dict(schema_version=1, campaign_id="qpt_quiroga_tuning_20261008_screen",
            phase="calibration",
            source=dict(commit=args.commit,
                        repo="/gpfs/workdir/silvetian/frankwolfe-quiroga-" + args.commit),
            shared=dict(n_qubits=2, rank=1, active_rows=128, observation_mode="noiseless",
                        initialization_basis="gell-mann",
                        initialization_convention="hs_unit_gell_mann_identity_sym_antisym_diag_v1",
                        initialization_seed_rule="200000+10*channel",
                        sampling_seed_rule="300000+100*channel",
                        fixed_penalty="lambda=128*alpha; constant throughout each trial",
                        printed_spectral_norm="dense exact Hermitian eigenspectrum",
                        exact_line_stopping=dict(full_gradient_atol=1e-12, full_gradient_rtol=1e-10),
                        work="Printed128 rows/update; exact ray counts direction, polynomial and candidate passes; checkpoints excluded"),
            gate=dict(fidelity_min=.99, rmse_max=1e-6, normalized_tp_max=.001),
            resources=dict(minimum_seconds_before_new_trial=120), trials=trials,
            qualification=("44 new literal-rule constant-multiplier/fixed-TP settings and six "
                           "separately labelled exact factor-ray line-search variants. The latter "
                           "keep the objective/direction but change the printed step; they are not "
                           "paper-exact adaFGD. No Armijo substitution, TP projection, or gate change."),
            preserved_prior_audit=dict(
                path="/home/tony/qpt_gell_mann_init_20261007_1750/audit_complete.json",
                sha256="73ae52860439868eb12fc59be9dc9f4c15d810e9463e0dc17b0811645a32fdd3"),
            selection_rule=("Literal rule and variant ranked separately by worst-channel deficit "
                            "max(RMSE/1e-6,normalizedTP/1e-3,max(0,1-fidelity)/.01). "
                            "Any failed or nonfinite case makes a setting ineligible for success. "
                            "A candidate must pass every calibration endpoint; tie-break by smaller "
                            "total charged rows, smaller alpha, then smaller multiplier. "
                            "Previous c1 pairs may be compared at their archived10k checkpoint only, "
                            "with prioraudit/checkpoint recorded. A passing candidate is frozen before "
                            "confirmation2..7; no retuning on confirmation or historicalrevalidation100..107. "
                            "If literalnonepass, optionally extend atmost2bestfinite settings to100k "
                            "only when trajectory diagnostics justify it; no unlimited blind extension. "
                            "Finalheldout1000..1019 untouched."))
assert len(trials) == 50
out = Path(__file__).with_name("stage") / "screen_draft_pinned.json"
with out.open("x") as stream:
    json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
print(out)
