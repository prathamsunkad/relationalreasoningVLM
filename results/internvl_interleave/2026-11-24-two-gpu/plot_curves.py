import csv
import gzip
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

bundle = Path(__file__).resolve().parent
provenance = json.loads((bundle/'PROVENANCE.json').read_text())
snapshot = {'checked_utc': provenance['captured_utc']}
root = bundle/'records'
out = bundle/'figures'
out.mkdir(exist_ok=True)
with gzip.open(root/'epoch6/learning-curve.jsonl.gz', 'rt') as f:
    train = [json.loads(line) for line in f]
with gzip.open(root/'epoch6/validation-curve.jsonl.gz', 'rt') as f:
    val = [json.loads(line) for line in f]
assert all(row['complete'] and row['count'] == 1000 for row in val)
assert [r['update'] for r in train] == list(range(1, train[-1]['update']+1))
checked = datetime.fromisoformat(snapshot['checked_utc']).astimezone(ZoneInfo('America/New_York'))
stamp = checked.strftime('%Y%m%d-%H%M')
base = out/f'epoch6-loss-accuracy-{stamp}'
window = 100
x = np.array([r['examples'] for r in train])/1000
vx = np.array([r['update']*16 for r in val])/1000
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.titlesize':15,
                     'axes.labelsize':11, 'axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#b5bdc9','text.color':'#17283c','axes.labelcolor':'#17283c',
                     'xtick.color':'#435268','ytick.color':'#435268','savefig.facecolor':'white'})
fig, axes = plt.subplots(1,2,figsize=(13,5.6))
fig.subplots_adjust(left=.07,right=.975,bottom=.25,top=.72,wspace=.24)
fig.text(.07,.95,'InternVL3.5 + Abstractor interleaving',fontsize=20,weight='bold',ha='left')
fig.text(.07,.902,f'Starting from epoch-6 LoRA  |  {train[-1]["examples"]:,} / 240,000 questions  |  {checked.strftime("Nov %d, %I:%M %p")} Eastern',fontsize=11,color='#566579')
train_color, val_color = '#8b9aab', '#196bc2'
legend = [Line2D([0],[0],color=train_color,lw=2,label='Training · trailing 100-update mean (1,600 questions)'),
          Line2D([0],[0],color=val_color,lw=2.3,marker='o',label='Validation · same 1,000 questions, unsmoothed')]
fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.065,.864),frameon=False,ncol=2,fontsize=10)
for ax,key,title in zip(axes,['loss','accuracy'],['Cross-entropy loss','Answer accuracy']):
    ax.axvspan(0,8,color='#f0e9fa',alpha=.9,zorder=0)
    ax.axvline(8,color='#9563be',linestyle='--',lw=1.15)
    raw=np.array([r[key] for r in train])
    factor=100 if key=='accuracy' else 1
    smoothed=np.convolve(raw,np.ones(window)/window,mode='valid')*factor
    ax.plot(x[window-1:],smoothed,color=train_color,lw=1.5,zorder=2)
    vv=np.array([r[key] for r in val])*factor
    ax.plot(vx,vv,color=val_color,lw=2.3,marker='o',ms=5,zorder=3)
    ax.set_title(title,loc='left',pad=12,weight='bold')
    ax.set_xlabel('Training questions processed (thousands)',labelpad=9)
    xmax = max(52, np.ceil(x[-1]/8)*8 + 2)
    ax.set_xlim(0,xmax)
    tick_step = 8 if xmax <= 90 else 16 if xmax <= 160 else 32
    ax.set_xticks(np.arange(0,xmax,tick_step))
    ax.grid(axis='y',color='#e8edf2',lw=.8)
    ax.set_axisbelow(True)
    ax.annotate(f'{vv[-1]:.1f}%' if key=='accuracy' else f'{vv[-1]:.3f}',
                (vx[-1],vv[-1]),xytext=(0,12),textcoords='offset points',ha='center',
                fontsize=11,color=val_color,weight='bold')
    if key=='accuracy':
        ax.set_ylim(50,86)
        ax.yaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    else:
        ax.set_ylim(.4,1.2)
        i=int(np.argmin(vv))
        ax.scatter([vx[i]],[vv[i]],s=95,facecolors='white',edgecolors=val_color,linewidth=1.6,zorder=4)
        ax.annotate(f'Lowest validation loss: {vv[i]:.3f}',(vx[i],vv[i]),
                    xytext=(max(11,vx[i]-20),.86),textcoords='data',fontsize=10,color=val_color,
                    arrowprops={'arrowstyle':'-','color':val_color,'lw':.9})
fig.text(.07,.125,'Shaded region: connector only (first 8,000 questions). Afterward: connector + LoRA; base VLM and Abstractor stay frozen.',fontsize=10,color='#435268')
fig.text(.07,.083,'Step 0 includes the untrained connector; it is not a LoRA-only baseline. Validation is measured every 8,000 questions.',fontsize=10,color='#435268')
fig.text(.07,.041,'Training uses different questions and training mode; its smoothed metrics are not directly comparable to held-out validation.',fontsize=9.5,color='#657387')
fig.savefig(base.with_suffix('.png'),dpi=180)
fig.savefig(base.with_suffix('.pdf'))
plt.close(fig)
with base.with_suffix('.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,lineterminator='\n',fieldnames=['update','training_questions','validation_loss','validation_accuracy'])
    writer.writeheader()
    writer.writerows({'update':r['update'],'training_questions':r['update']*16,'validation_loss':r['loss'],'validation_accuracy':r['accuracy']} for r in val)
base.with_suffix('.json').write_text(json.dumps({'snapshot':'records','checked_utc':snapshot['checked_utc'],'training_smoothing_updates':window,'training_points':len(train),'validation_points':len(val)},indent=2)+'\n')
print(json.dumps({'png':str(base.with_suffix('.png')),'pdf':str(base.with_suffix('.pdf')),'csv':str(base.with_suffix('.csv'))},indent=2))
