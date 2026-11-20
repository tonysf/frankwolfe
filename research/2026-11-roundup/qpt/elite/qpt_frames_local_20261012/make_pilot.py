"""Freeze the small original-backend timing/parity reference before execution."""
from run_local import ROOT,COMMIT,source,write,now
trial=dict(id='reference-c0-2000',channel=0,alpha0=5e-5,penalty_exponent=.25,
           step_scale=4.,step_offset=4.,step_exponent=1.,rho_scale=2.,rho_offset=4.,rho_exponent=.6,
           tau=2.,batch_size=32,steps=2000,metrics_every=100,sampling_replicate=0)
write(ROOT/'plan_pilot.json',dict(scope='local_n2_full_noiseless_frames_calibration',phase='pilot',
    source_commit=COMMIT,source_sha256=source(),created_utc=now(),
    gate={'fidelity_min':.99,'rmse_max':1e-6,'normalized_tp_max':.001},trials=[trial],
    role='Timing and unchanged-runner reference only; no selected setting.'))
