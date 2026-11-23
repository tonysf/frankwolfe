"""Small metadata-only fixtures for the full-population report dimension."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('full_shareseq_report', HERE/'report.py')
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def write(path, value):
    path.write_text(json.dumps(value))


def build(directory):
    n = 31890
    config = json.loads((HERE/'campaign.json').read_text())
    config.update(expected_n=n, instance_id='shareseq_full', scope='Exploratory full retained SHARE-seq population')
    hashes = {name+'.npy':'a'*64 for name in ('A','B','a','b')}
    manifest = dict(n=n,m=n,id='shareseq_full',array_files_sha256=hashes)
    frozen = dict(config=config,instance_manifest=manifest,slurm_job_id='fixture-job',
                  source_manifest_sha256='b'*64,correspondence_sha256='c'*64,source=str(HERE))
    write(directory/'frozen_campaign.json',frozen)
    write(directory/'correctness.json',dict(status='passed'))
    write(directory/'pilot_gate.json',dict(status='passed'))
    results = []
    for seed in (500,501,502):
        for method_index,method in enumerate(reporter.METHODS):
            folder = directory/f'final_{seed}_{method}'
            folder.mkdir()
            solver,updates = 300.1,1000+seed
            cfg = dict(initialization='product',init_seed=100,batch=128,rho0=.3,gamma0=.1,
                       mu_scale=.1,eta=.1,inner_tol=.0001,inner_max=1000,budget=300,
                       max_updates=0,device='cuda',save_plans=True,method=method,seed=seed,
                       checkpoints='10,30,100,300',correspondence_sha256='c'*64)
            env = dict(world_size=4,backend='nccl',slurm_job_id='fixture-job',
                       deterministic_algorithms=True,tf32=False)
            timing = dict(solver_seconds=solver,solver_gpu_seconds=4*solver,
                          total_elapsed_seconds=350,repair_seconds=.5)
            counters = dict(completed_updates=updates,inner_stopping_failures=0,inner_iterations_total=0)
            trace = []
            for index,target in enumerate((None,10,30,100,300)):
                iteration = 0 if target is None else int(updates*target/300)
                elapsed = .01 if target is None else target+.1
                loss = .2 if target is None else .1+.02/(index+1)+.001*method_index+.00001*(seed-500)
                trace.append(dict(kind='initial' if target is None else 'checkpoint',
                    target_solver_seconds=target,actual_solver_seconds=elapsed,iteration=iteration,
                    raw_loss=loss+.001,rounded_loss=loss,repair_seconds_current=.1,
                    time_to_this_feasible_output=elapsed+.1,rounded_both_l1=1e-8,
                    correspondence_top1_accuracy=1/n,correspondence_rank_fraction=.5,
                    correspondence_true_pair_mass=1/n))
            validation = dict(dtype='float64',exact_objective=True,rounded_loss=trace[-1]['rounded_loss'],
                              raw_loss=.11,rounded=dict(both_l1=1e-8))
            memory = dict(cuda_peak_reserved_bytes=20*1024**3,cuda_peak_allocated_bytes=18*1024**3,
                          cuda_total_memory_bytes=40*1024**3,cpu_process_peak_rss_kib=1000000)
            ranks = [dict(rank=i,gpu='NVIDIA A100-SXM4-40GB',
                          memory={phase:memory.copy() for phase in ('preparation','solver_and_diagnostics','float64_validation')},
                          saved_plans={kind:dict(shape=[n//4+int(i<n%4),n],file='/fixture/'+kind,
                                                file_sha256='d'*64) for kind in ('P_raw','P_rounded')})
                     for i in range(4)]
            run = dict(status='completed',method=reporter.method_names()[method],config=cfg,environment=env,
                       array_files_sha256=hashes,instance_manifest=manifest,
                       source_file_sha256={'run_sharded.py':'e'*64},counters=counters,timing=timing,
                       final_float64_validation=validation,trace=trace,ranks=ranks)
            write(folder/'metadata.json',run)
            results.append(dict(phase='final',seed=seed,method=method,path=str(folder),status='completed',
                                updates=updates,loss=validation['rounded_loss'],timing=timing))
    write(directory/'completion.json',dict(status='completed',results=results,allocated_workload_seconds=2200))
    return frozen


def test_full_dimension_generates_all_reports_and_uneven_shards(tmp_path, monkeypatch):
    import matplotlib.figure
    build(tmp_path)
    titles = []
    original = matplotlib.figure.Figure.savefig

    def capture_title(figure,*args,**kwargs):
        titles.append(figure.axes[0].get_title())
        return original(figure,*args,**kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure,'savefig',capture_title)
    result = reporter.generate(tmp_path)
    assert result['valid_final_runs'] == 6 and len(result['figures']) == 4
    assert len(titles) == 4 and all('31,890 paired cells' in title for title in titles)
    assert result['uniform_baselines'] == dict(top1_accuracy=1/31890,normalized_rank=.5,true_pair_mass=1/31890)
    assert result['provenance']['instance_manifest']['id'] == 'shareseq_full'
    assert all((tmp_path/item['file']).is_file() for item in result['figures'])
    text = (tmp_path/'report.md').read_text()
    assert '# SHARE-seq 31,890-cell' in text
    assert '1/31,890' in text and 'divided by 31,889' in text
    assert 'approximately 48.8 GB' in text
    assert '25,000' not in text and '24,999' not in text
    assert result['aggregate']['methods']['smoothing-fw']['summaries']['completed_outer_updates']['median'] == 1501


@pytest.mark.parametrize('change', ('missing_n','missing_expected_n','wrong_n','wrong_m','wrong_id','missing_id','bool_n'))
def test_invalid_frozen_dimension_fails_before_report_publication(tmp_path,change):
    frozen = build(tmp_path)
    if change == 'missing_n':
        del frozen['instance_manifest']['n']
    elif change == 'missing_expected_n':
        del frozen['config']['expected_n']
    elif change == 'wrong_n':
        frozen['config']['expected_n'] = 25000
    elif change == 'wrong_m':
        frozen['instance_manifest']['m'] = 25000
    elif change == 'wrong_id':
        frozen['instance_manifest']['id'] = 'shareseq25000'
    elif change == 'missing_id':
        del frozen['config']['instance_id']
    elif change == 'bool_n':
        frozen['instance_manifest']['n'] = True
    write(tmp_path/'frozen_campaign.json',frozen)
    with pytest.raises(ValueError):
        reporter.generate(tmp_path)
    assert not (tmp_path/'report.json').exists()
    assert not (tmp_path/'report.md').exists()
    assert not list(tmp_path.glob('*.png'))


def test_saved_shard_shape_uses_full_uneven_partition(tmp_path):
    build(tmp_path)
    path = tmp_path/'final_500_smoothing-fw'/'metadata.json'
    run = json.loads(path.read_text())
    assert [rank['saved_plans']['P_raw']['shape'][0] for rank in run['ranks']] == [7973,7973,7972,7972]
    run['ranks'][0]['saved_plans']['P_raw']['shape'] = [6250,25000]
    write(path,run)
    with pytest.raises(ValueError,match='saved shard'):
        reporter.audit(tmp_path)


def test_missing_checkpoint_remains_missing_for_full_population(tmp_path):
    build(tmp_path)
    path = tmp_path/'final_500_smoothing-fw'/'metadata.json'
    run = json.loads(path.read_text())
    run['trace'][2] = dict(kind='missed',target_solver_seconds=30,actual_solver_seconds=30.1,
                           iteration=150,reason='same_completed_iterate_used_for_earlier_target')
    write(path,run)
    evidence,runs,names = reporter.audit(tmp_path)
    point = reporter.aggregate(evidence,runs,names)['methods']['smoothing-fw']['checkpoints']['30.0']
    assert point['n'] == 2 and point['summary'] is None and point['missing_seeds'] == [500]
