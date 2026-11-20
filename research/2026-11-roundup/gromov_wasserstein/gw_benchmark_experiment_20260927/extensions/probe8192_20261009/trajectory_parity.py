"""Pure-metadata parity proof for optional diagnostics; no scientific imports."""
from __future__ import annotations


def compare_trajectories(reference, observed):
    def require(condition, message):
        if not condition:
            raise ValueError('Trajectory parity failed: '+message)
    require(reference['status'] == observed['status'] == 'completed', 'incomplete run')
    reference_config = {k:v for k,v in reference['config'].items() if k not in ('out','direction_diagnostics')}
    observed_config = {k:v for k,v in observed['config'].items() if k not in ('out','direction_diagnostics')}
    require(reference_config == observed_config, 'optimizer/input configuration differs')
    require(reference['dimensions'] == observed['dimensions'], 'dimensions differ')
    require(reference['array_files_sha256'] == observed['array_files_sha256'], 'input arrays differ')
    require(reference['instance_manifest_sha256'] == observed['instance_manifest_sha256'], 'input manifest differs')
    common_sources = set(reference['source_file_sha256']) & set(observed['source_file_sha256'])
    require(all(reference['source_file_sha256'][k] == observed['source_file_sha256'][k] for k in common_sources),
            'shared numerical sources differ')
    require(reference['counters'] == observed['counters'], 'oracle/update counters differ')
    require(reference['final_float64_validation'] == observed['final_float64_validation'], 'FP64 output validation differs')
    require(reference['output_selection'] == observed['output_selection'], 'independent sampled output differs')
    require(len(reference['ranks']) == len(observed['ranks']), 'rank counts differ')
    hashes = []
    for a,b in zip(reference['ranks'],observed['ranks']):
        require(a['rank'] == b['rank'], 'rank order differs')
        require(a['initial_shard_sha256'] == b['initial_shard_sha256'], 'initial coupling differs')
        ah = {k:v['array_sha256'] for k,v in a['saved_plans'].items()}
        bh = {k:v['array_sha256'] for k,v in b['saved_plans'].items()}
        require(ah == bh and set(ah) == {'selected_raw','selected_rounded','last_raw','last_rounded'}, 'output plan hashes differ')
        hashes.append(dict(rank=a['rank'], initial_shard_sha256=a['initial_shard_sha256'], plan_array_sha256=ah))
    require(len(reference['trace']) == len(observed['trace']), 'diagnostic trace lengths differ')
    for a,b in zip(reference['trace'],observed['trace']):
        for field in ('kind','iteration','raw_loss','rounded_loss','raw_residuals','rounded_residuals','repair_l1'):
            require(a[field] == b[field], 'diagnostic trace differs: '+field)
    return dict(status='passed', batch=observed['config']['batch'], iterations=observed['config']['iterations'],
                counters=observed['counters'], ranks=hashes,
                final_float64_validation=observed['final_float64_validation'],
                checked_diagnostic_rows=len(observed['trace']),
                scope='Exact coupling hashes, objective/residual values and counters; execution timings intentionally differ.')
