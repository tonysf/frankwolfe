#!/bin/bash
#SBATCH --job-name=gw_msbpg_check
#SBATCH --account=fwllm
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --exclude=ruche-gpu16,ruche-gpu19
#SBATCH --export=NONE
#SBATCH --propagate=NONE
#SBATCH --output=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/msbpg-validation-%j.out
set -euo pipefail
GW_SOURCE=${1:?immutable source root}
GW_PREPARED=${2:?existing prepared dataset root}
GW_OUTPUT=${3:?new validation output directory}
cd "$GW_SOURCE"
sha256sum -c SHA256SUMS
module purge
module load miniconda3/25.5.1/none-none
source activate /gpfs/workdir/silvetian/.conda/envs/fineweb
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export SLURM_EXPORT_ENV=ALL
GW_PYTHON=/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/.venv/bin/python
GW_EXTENSION="$GW_SOURCE/extensions/msbpg_20261005"
mkdir -p "$GW_OUTPUT"
trap 'rc=$?; printf "%s\n" "$rc" > "$GW_OUTPUT/exit_code.txt"' EXIT
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$GW_OUTPUT/gpus.csv"
"$GW_PYTHON" -m pip freeze > "$GW_OUTPUT/environment.lock.txt"
"$GW_PYTHON" -m pytest -q "$GW_EXTENSION" > "$GW_OUTPUT/cpu_tests.txt" 2>&1
"$GW_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 \
    "$GW_EXTENSION/validate_distributed.py" --device cuda --out "$GW_OUTPUT/distributed_cuda.json"
# Actual-size, actual-topology MSBPG smokes follow kernel correctness.
# Keep no dense plans; hashes and independent FP64 objective checks are retained.
for GW_START in product randomized; do
  "$GW_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 \
      "$GW_EXTENSION/run_msbpg.py" --device cuda --instance "$GW_PREPARED/shareseq_full" \
      --method published-msbpg --iterations 3 --batch 128 --seed 610 --output-seed 910 \
      --initialization "$GW_START" --init-seed 10610 --coupling-fraction 1 --momentum-prefactor 1 \
      --dtype float32 --save-plans none --out "$GW_OUTPUT/shareseq_full_${GW_START}_smoke"
done
for GW_INSTANCE in bmmc6224 snare_full; do
  for GW_START in product randomized; do
    srun --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=gpu:1 --exact \
        "$GW_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=1 \
        "$GW_EXTENSION/run_msbpg.py" --device cuda --instance "$GW_PREPARED/$GW_INSTANCE" \
        --method published-msbpg --iterations 3 --batch 128 --seed 610 --output-seed 910 \
        --initialization "$GW_START" --init-seed 10610 --coupling-fraction 1 --momentum-prefactor 1 \
        --dtype float32 --save-plans none --out "$GW_OUTPUT/${GW_INSTANCE}_${GW_START}_smoke"
  done
done
# Gate the smallest allowed step at the last production schedule index as well.
# These representative-state checks do not pretend to be 2000-update trajectories.
"$GW_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 \
    "$GW_EXTENSION/validate_late_steps.py" --device cuda --instance "$GW_PREPARED/shareseq_full" \
    --horizon 2000 --coupling-fraction .125 --momentum-prefactor .1 --out "$GW_OUTPUT/shareseq_full_late.json"
for GW_INSTANCE in bmmc6224 snare_full; do
  srun --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=gpu:1 --exact \
      "$GW_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=1 \
      "$GW_EXTENSION/validate_late_steps.py" --device cuda --instance "$GW_PREPARED/$GW_INSTANCE" \
      --horizon 2000 --coupling-fraction .125 --momentum-prefactor .1 --out "$GW_OUTPUT/${GW_INSTANCE}_late.json"
done
"$GW_PYTHON" - "$GW_EXTENSION" "$GW_PREPARED" "$GW_OUTPUT" <<'PY_CHECK'
import importlib.util, json, math, pathlib, sys
extension, prepared, output = map(pathlib.Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location('msbpg_gate_contract', extension/'campaign.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
config = module.validate_config(module.read_json(extension/'campaign.json'))
module.load_baseline(extension/'campaign.json', config)
receipts = []
late_receipts = []
for instance in config['instances']:
    provenance = module.identity(extension/'campaign.json', extension.parents[1], prepared, instance)
    for start in config['initializations']:
        path = output/f'{instance}_{start}_smoke'/'metadata.json'
        metadata = module.read_json(path)
        module.validate_metadata(metadata, metadata['config'], provenance, config['instances'][instance],
                                 config['final_marginal_l1_tolerance'])
        if metadata['config']['iterations'] != 3 or metadata['config']['save_plans'] != 'none':
            raise ValueError('Incorrect smoke configuration')
        receipts.append(dict(instance=instance, initialization=start, metadata_sha256=module.sha256(path)))
    late_path = output/f'{instance}_late.json'
    late = module.read_json(late_path)
    assert late['status'] == 'passed' and late['horizon'] == 2000 and late['iteration'] == 1999
    assert late['coupling_fraction'] == .125 and late['momentum_prefactor'] == .1
    assert late['world_size'] == config['instances'][instance]['world_size']
    assert late['dimensions'] == dict(source=config['instances'][instance]['n'], target=config['instances'][instance]['n'])
    for key in ('array_files_sha256', 'instance_manifest_sha256', 'source_file_sha256'):
        assert late[key] == provenance[key], 'Late-step provenance mismatch: '+key
    assert late['validation_driver_sha256'] == module.sha256(extension/'validate_late_steps.py')
    assert len(late['checks']) == 2 and {x['initialization'] for x in late['checks']} == {'product', 'randomized'}
    for check in late['checks']:
        assert check['status'] == 'passed' and check['iteration'] == 1999
        cert = check['certificate']
        assert cert['converged'] is True and cert['proximal_optimality_certified'] is True and cert['certificate_valid'] is True
        for key in ('marginal_l1', 'normalized_gap_upper_bound'):
            assert 0 <= cert[key] <= 1e-7 and math.isfinite(cert[key]), key
        assert 0 <= cert['primal_witness_marginal_l1'] <= 1e-12
        constants = late['theorem_constants']
        alpha = .125/(2*math.sqrt(2*constants['hessian_bound']**2))*.1/math.sqrt(2000)
        assert math.isclose(cert['alpha'], alpha, rel_tol=1e-12)
        assert math.isclose(cert['eta'], alpha/constants['kernel_scale'], rel_tol=1e-12)
        for key in ('retained_log_normalized_displacement_squared', 'stored_plan_normalized_displacement_squared'):
            assert math.isfinite(check[key]) and check[key] >= 0
    late_receipts.append(dict(instance=instance, metadata_sha256=module.sha256(late_path)))
module.write_json(output/'gate.json', dict(status='completed', smoke_receipts=receipts, late_step_receipts=late_receipts,
                  scope='Four-rank correctness; actual-size 1/1/4-A100 smoke checks; smallest-prefactor final-index numerical gates at both starts.'))
PY_CHECK
