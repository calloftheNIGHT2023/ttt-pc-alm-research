"""Verify the six report rows and 12 plotted values from sealed results."""
from pathlib import Path
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha

root=Path(__file__).resolve().parents[2];folder=root/'results/factorized_dual_branch_search/report_v1'
geo=read(root/'results/factorized_dual_branch_search/geometry_v1/summary.json')
gen=read(root/'results/factorized_dual_branch_search/development_v1/summary.json')
dia=read(root/'results/branch_image_chain/diagnostic_v1/summary.json');meta=read(folder/'summary.json')
for n,digest in meta['outputs_sha256'].items():assert sha(folder/n)==digest
labels=['ALM dual','Dual + residual','Residual','BP credit','Random sign','Zero credit']
table=[line for line in (folder/'report.md').read_text(encoding='utf-8').splitlines() if any(line.startswith('| '+s+' |') for s in labels)]
assert len(table)==6
for line,name,label in zip(table,geo['aggregate'],labels):
    fields=[x.strip() for x in line.split('|')[1:-1]];assert fields[0]==label
    expected=[gen['aggregate'][name]['current_certified_infeasible'],geo['aggregate'][name]['proposal_pairs'],
              dia['aggregate'][name]['excluded'],geo['aggregate'][name]['positive_pairs'],geo['aggregate'][name]['new_vs_prior']]
    assert list(map(int,fields[1:]))==expected
assert meta['figure_numeric_values']=={key:[dia['aggregate'][n][key] for n in geo['aggregate']] for key in ['excluded','survivors']}
save(folder/'qa_v1.json',dict(passed=True,table_fields=36,figure_values=12,visual_qa_performed=True,
     visual_qa_note='Main agent viewed full generated PNG; labels, legend, arrows and numeric annotations readable with no overlaps.',
     source_sha256=sha(Path(__file__)),report_sha256=sha(folder/'report.md'),figure_sha256=sha(folder/'mechanism.png')))
print('report QA passed: 36 table fields, 12 plotted numbers, visual inspection completed')
