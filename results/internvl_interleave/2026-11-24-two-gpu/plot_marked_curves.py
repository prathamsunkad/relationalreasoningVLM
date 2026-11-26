"""Plot archived training and validation; mark best and latest checkpoints.

Run from any directory; requires matplotlib and numpy, no models or GPU.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

bundle = Path(__file__).resolve().parent
provenance = json.loads((bundle/'PROVENANCE.json').read_text())
root = bundle
snapshot = bundle/'records'
def rows(name):
    import gzip
    with gzip.open(snapshot/'epoch6'/(name+'.gz'), 'rt') as handle:
        return [json.loads(line) for line in handle if line.strip()]
train = rows('learning-curve.jsonl')
val = [r for r in rows('validation-curve.jsonl') if r['complete'] and r['count'] == 1000]
assert [r['update'] for r in train] == list(range(1, train[-1]['update']+1))
assert len({r['update'] for r in val}) == len(val)
best_loss = min(val, key=lambda r:r['loss'])
best_acc = max(val, key=lambda r:r['accuracy'])
latest = val[-1]
end = train[-1]['update']
checked = datetime.fromisoformat(provenance['captured_utc']).astimezone(ZoneInfo('America/New_York'))
out = root/'figures'
out.mkdir(exist_ok=True)
stem = out/f'progress-marked-{checked.strftime("%Y%m%d-%H%M")}'

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':14,
                     'axes.labelsize':11,'axes.edgecolor':'#cbd5e1','axes.labelcolor':'#334155',
                     'xtick.color':'#64748b','ytick.color':'#64748b','savefig.facecolor':'white'})
fig, axes = plt.subplots(2,2,figsize=(14,9.4),sharex=True,
                         gridspec_kw={'height_ratios':[1.2,1]})
fig.subplots_adjust(left=.075,right=.965,top=.81,bottom=.145,hspace=.37,wspace=.20)
fig.text(.075,.962,'InternVL interleaving: training progress',fontsize=23,weight='bold',color='#142c45')
fig.text(.075,.923,'Starting from epoch-6 LoRA  ·  2 H200 GPUs  ·  Frozen base VLM and Abstractor',
         fontsize=12,color='#475569')
fig.text(.075,.887,f'{end:,} updates / {train[-1]["examples"]:,} questions  '
         f'({100*train[-1]["examples"]/240000:.1f}% of the planned 240k pass)  ·  '
         f'Snapshot: {checked.strftime("%b %d, %I:%M %p")} Eastern',fontsize=11,color='#475569')
colors={'val':'#6d4cc4','train':'#167d8d','best':'#007f59','latest':'#ce6414'}
x=np.array([r['update'] for r in val])
formatter=FuncFormatter(lambda n,pos:f'{int(n):,}')
for ax in axes.flat:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#e8edf2',zorder=0)
    ax.set_xlim(0,end+330)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7,integer=True))
    ax.xaxis.set_major_formatter(formatter)
    ax.axvspan(0,500,color='#e2e8f0',alpha=.65,zorder=0)
    ax.axvline(end,color='#94a3b8',lw=1.1,ls='--',zorder=0)

def annotate(ax,row,key,text,offset,align='left'):
    value=row[key]*(100 if key=='accuracy' else 1)
    best=text.startswith('Best')
    col=colors['best' if best else 'latest']
    ax.scatter([row['update']],[value],s=140 if best else 75,
               marker='*' if best else 'D',color=col,edgecolor='white',linewidth=.9,zorder=5)
    ax.annotate(text,(row['update'],value),xytext=offset,textcoords='offset points',
                color=col,fontsize=11,weight='bold',ha=align,va='center',
                bbox={'boxstyle':'round,pad=.32','fc':'white','ec':'none','alpha':.96},
                arrowprops={'arrowstyle':'-','color':col,'lw':1.1},zorder=6)

ax=axes[0,0]
ax.set_title('Validation loss',loc='left',weight='bold',pad=12)
ax.plot(x,[r['loss'] for r in val],'-o',color=colors['val'],lw=2.2,ms=4,zorder=3)
ax.set_ylabel('Candidate cross-entropy')
ax.set_ylim(.968,1.145)
annotate(ax,best_loss,'loss',f'Best loss: {best_loss["loss"]:.4f}\nStep {best_loss["update"]:,}',(-10,-35),'center')
annotate(ax,latest,'loss',f'Latest: {latest["loss"]:.4f}\nStep {latest["update"]:,}',(-13,-34),'right')

ax=axes[0,1]
ax.set_title('Validation accuracy',loc='left',weight='bold',pad=12)
ax.plot(x,[100*r['accuracy'] for r in val],'-o',color=colors['val'],lw=2.2,ms=4,zorder=3)
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(58.2,69.7)
annotate(ax,best_acc,'accuracy',f'Best accuracy: {100*best_acc["accuracy"]:.1f}%\nStep {best_acc["update"]:,}',(3,33))
annotate(ax,latest,'accuracy',f'Latest: {100*latest["accuracy"]:.1f}%\nStep {latest["update"]:,}',(-12,-44),'right')

tx=np.array([r['update'] for r in train])
window=100
def rolling(values):
    v=np.array(values,dtype=float);s=np.concatenate(([0.],np.cumsum(v)))
    ends=np.arange(1,len(v)+1);starts=np.maximum(0,ends-window)
    return (s[ends]-s[starts])/(ends-starts)
for j,(key,label) in enumerate([('loss','Candidate cross-entropy'),('accuracy','Accuracy (%)')]):
    ax=axes[1,j]
    y=rolling([r[key] for r in train])*(100 if key=='accuracy' else 1)
    ax.plot(tx,y,color=colors['train'],lw=1.8)
    ax.set_title('Training '+('loss' if key=='loss' else 'accuracy')+' · 100-update mean',loc='left',weight='bold',pad=12)
    ax.set_ylabel(label)
    ax.set_xlabel('Optimizer updates (16 questions per update)')
    ax.scatter(tx[-1],y[-1],s=30,color=colors['train'],zorder=3)

fig.text(.075,.061,'Validation: all 1,000 fixed questions, unsmoothed. Green star = best; orange diamond = latest validation.',fontsize=10,color='#475569')
fig.text(.075,.034,'Shaded steps 0–500: connector only; thereafter connector + LoRA. Dashed line: latest training update. Checkpoint selection uses minimum validation loss.',fontsize=9.5,color='#475569')
fig.savefig(str(stem)+'.png',dpi=170)
fig.savefig(str(stem)+'.pdf')
payload={'snapshot':'records','captured_utc':provenance['captured_utc'],'validation_count':1000,'training_update':end,'training_examples':train[-1]['examples'],
         'best_validation_loss':best_loss,'best_validation_accuracy':best_acc,'latest_validation':latest}
Path(str(stem)+'.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps({'png':str(stem)+'.png','pdf':str(stem)+'.pdf','best_loss':best_loss,'best_accuracy':best_acc,'latest':latest}))
