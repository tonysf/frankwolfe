"""Adversarial allocation-accounting cases; no network, Slurm or GPU work."""
import csv
import io
import pytest
from account_compute_resources import FIELDS, summarize


def ledger(*jobs):
    return {'jobs':[{'job_id':i,'purpose':'test',**extra} for i,extra in jobs]}


def row(identity,state='COMPLETED',elapsed='10',tres='cpu=2,gres/gpu=1',partition='gpua100',start='2026-09-28T01:00:00'):
    d=dict.fromkeys(FIELDS,'')
    d.update(JobID=identity,JobIDRaw=identity,State=state,ElapsedRaw=elapsed,AllocTRES=tres,Partition=partition,Start=start)
    return d


def raw(*rows):
    out=io.StringIO();w=csv.DictWriter(out,FIELDS,delimiter='|',lineterminator='\n');w.writeheader();w.writerows(rows)
    return out.getvalue()


def test_primary_rows_steps_and_array_parent_do_not_double_count():
    d=ledger((10,{'array_tasks':2}),(20,{}))
    r=summarize(d,raw(row('10',elapsed='999'),row('10_0',elapsed='7'),row('10_0.batch',elapsed='777'),
                      row('10_0.extern',elapsed='888'),row('10_0.0',elapsed='999'),row('10_1',elapsed='9'),
                      row('20','FAILED',elapsed='5')))
    assert r['accounting_complete'] and r['known_snapshot_a100_gpu_seconds']==21
    assert r['excluded_row_counts']=={'aggregate_array_parent':1,'job_step':3}


def test_typed_gpu_tres_is_not_added_to_total():
    r=summarize(ledger((10,{})),raw(row('10',elapsed='7',tres='cpu=2,gres/gpu=2,gres/gpu:a100=2')))
    assert r['accounting_complete'] and r['known_snapshot_a100_gpu_seconds']==14
    broken=summarize(ledger((10,{})),raw(row('10',tres='gres/gpu=2,gres/gpu:a100=1')))
    assert not broken['snapshot_coverage_complete'] and broken['known_snapshot_a100_gpu_seconds']==0


def test_cpu_and_cancelled_unstarted_zero_are_not_gpu_time():
    r=summarize(ledger((10,{}),(20,{})),raw(row('10',elapsed='20',tres='cpu=2,mem=8G',partition='cpu_short'),
        row('20','CANCELLED by 3359','0','','gpua100','None')))
    assert r['accounting_complete'] and r['known_snapshot_a100_gpu_seconds']==0
    assert r['cpu_only_primary_rows']==1 and r['cpu_only_elapsed_seconds']==20


def test_active_and_compressed_pending_are_complete_snapshot_not_final():
    r=summarize(ledger((10,{'array_tasks':5})),raw(row('10_0'),row('10_1','RUNNING',elapsed='3'),
        row('10_[2-4%2]','PENDING','0','','gpua100','Unknown')))
    assert r['snapshot_coverage_complete'] and not r['accounting_complete']
    assert r['terminal_a100_gpu_seconds']==10 and r['nonterminal_accrued_a100_gpu_seconds']==3
    assert r['allocation_states']=={'COMPLETED':1,'RUNNING':1,'PENDING':3}


def test_compressed_started_allocation_is_not_multiplied():
    r=summarize(ledger((10,{'array_tasks':2})),raw(row('10_[0-1]')))
    assert r['known_snapshot_a100_gpu_seconds']==0 and len(r['missing_primary_rows'])==2
    assert not r['accounting_complete'] and r['issues']


def test_partial_arrays_and_missing_gpu_allocation_fail_closed():
    r=summarize(ledger((10,{'array_task_indices':[3,7,11]})),raw(row('10_3'),row('10_7',tres='cpu=2')))
    assert r['known_snapshot_a100_gpu_seconds']==10 and not r['snapshot_coverage_complete']
    assert r['missing_primary_rows']==['10_11'] and r['jobs'][0]['unresolved_primary_rows']==['10_11','10_7']


def test_duplicate_and_out_of_scope_rows_prevent_complete_claim():
    r=summarize(ledger((10,{})),raw(row('10'),row('10'),row('99')))
    assert not r['accounting_complete'] and r['known_snapshot_a100_gpu_seconds']==0
    assert len(r['issues'])==2


@pytest.mark.parametrize('elapsed',['-1','NaN','1.5'])
def test_invalid_elapsed_is_not_silently_zero(elapsed):
    r=summarize(ledger((10,{})),raw(row('10',elapsed=elapsed)))
    assert r['accounted_primary_rows']==0 and r['issues']


@pytest.mark.parametrize('state',['COMPLETED','RUNNING'])
def test_report_renders_terminal_and_active_snapshots(state):
    from account_compute_resources import markdown
    r=summarize(ledger((10,{})),raw(row('10',state)))
    r['input_provenance']={'queried_from_utc':'a','queried_until_utc':'b',
        'snapshot_directory':'fixture','raw_sacct_sha256':'test','helper_sha256':'test'}
    text=markdown(r)
    assert ('not the final experiment total' in text)==(state=='RUNNING')
    assert ('All scoped allocations are terminal and accounted for.' in text)==(state=='COMPLETED')
    assert 'primary rows before compressed-array expansion: 1' in text
