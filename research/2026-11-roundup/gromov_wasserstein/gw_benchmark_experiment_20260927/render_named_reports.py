"""Rerender frozen result summaries with explicit names; never rerun optimizers."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil
import matplotlib
matplotlib.use('Agg')
import summarize_campaign as campaign
import summarize_momentum_replays as momentum
from method_names import METHOD_NAMES
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    sources={'campaign':ROOT/'reports/stage_c_final','momentum':ROOT/'reports/momentum_replay_completion'}
    targets={'campaign':ROOT/'reports/stage_c_final_named','momentum':ROOT/'reports/momentum_replay_named'}
    audit=dict(generated_utc=datetime.now(timezone.utc).isoformat(),scope='Presentation only. No numerical reruns, selection changes or new statistical aggregation.',names=METHOD_NAMES,rendering_source_sha256={p:sha(ROOT/p) for p in ('method_names.py','summarize_campaign.py','summarize_momentum_replays.py','render_named_reports.py')},reports={},figures=[])
    note='Method definitions: [precise names, gradient oracles and constraint updates](../METHOD_NAMES.md). Exact and full gradient meant the same unsampled gradient; all plotted objectives use the complete GW loss. The original report JSON/CSV is copied unchanged; updated figure/table labels are documented in [the presentation audit](../METHOD_NAME_REVISION.json).\n\n'
    for kind,source in sources.items():
        out=targets[kind];out.mkdir(exist_ok=False)
        r=json.loads((source/'report.json').read_text())
        for name in ('report.json','runs.csv','curve_summaries.csv'):
            if (source/name).exists():
                shutil.copyfile(source/name,out/name)
                assert sha(source/name)==sha(out/name)
        if kind=='campaign':
            figures=campaign.figures(r['across_problem_aggregates'],r['candidate_scores'],out,r['scope'])
            assert figures==r['figures']
            text=campaign.make_markdown(r)
        else:
            figures,curves=momentum.make_plots(r['runs'],out)
            assert figures==r['figures'] and curves==r['curve_summaries']
            text=(source/'report.md').read_text()
        first,rest=text.split('\n',1)
        (out/'report.md').write_text(first+'\n\n'+note+rest.lstrip('\n'))
        audit['reports'][kind]=dict(source=str(source.relative_to(ROOT)),presentation=str(out.relative_to(ROOT)),original_report_sha256=sha(source/'report.json'),presentation_report_sha256=sha(out/'report.json'),scientific_json_byte_identical=True,figure_count=len(figures))
        for name in figures:
            path=out/name
            if path.suffix=='.pdf':assert path.read_bytes().startswith(b'%PDF-')
            audit['figures'].append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
    (ROOT/'reports/METHOD_NAME_REVISION.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'status':'rendered','reports':audit['reports'],'figure_files':len(audit['figures'])},indent=2))

if __name__=='__main__':main()
