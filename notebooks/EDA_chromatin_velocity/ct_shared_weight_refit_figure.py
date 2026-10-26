#!/usr/bin/env python3
"""Compare both specified fit/tuning strategies on identical target cohorts."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs/ct_shared_weight_forecast_v1';FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none'})
primary=pd.read_csv(OUT/'model/pooled_metrics.csv')
models=['own_history','own_nonlinear','constant_cis','flexible_cis','distance_flexible','generic_smoothing']
labels=['Own history','Own nonlinear','Constant Cicero','Shared Cicero','Distance control','Generic smoothing']
fig,axes=plt.subplots(1,2,figsize=(170/25.4,135/25.4),sharey=True);fig.subplots_adjust(left=.23,right=.97,bottom=.30,top=.73,wspace=.24)
rows=[]
for ax,cohort,title in zip(axes,['total_variance_top20','gini_top20'],['High total variance','High Gini']):
 other=pd.read_csv(OUT/'model_selected_refit'/cohort/'pooled_metrics.csv')
 for di,(d,label,color) in enumerate([(primary,'Fit/tune on all eligible','#337d9f'),(other,'Fit/tune on selected','#cd792a')]):
  q=d[(d.cohort==cohort)&(d.interval=='pooled')&(d.error=='rate_sse')].set_index('model').loc[models]
  v=-100*q.vs_persistence.to_numpy();lo=-100*q.vs_persistence_high.to_numpy();hi=-100*q.vs_persistence_low.to_numpy();y=np.arange(len(models))+(di-.5)*.23
  ax.hlines(y,lo,hi,color=color,lw=1);ax.plot(v,y,'o',color=color,ms=3,label=label)
  for j,m in enumerate(models):rows.append(dict(cohort=cohort,fit_and_tuning_population='all_eligible' if di==0 else cohort,model=m,error_increase_percent=v[j],chromosome_bootstrap_low_percent=lo[j],chromosome_bootstrap_high_percent=hi[j]))
 ax.axvline(0,color='#555555',lw=.7);ax.set_title(title+'\nIdentical held-out target masks',pad=12);ax.set_xlabel('Error increase vs persistence (%)');ax.set_yticks(range(len(models)),labels);ax.set_ylim(5.6,-.6);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.1)
fig.suptitle('Does fitting specifically on variable peaks help?',x=.035,y=.97,ha='left',fontsize=11)
fig.text(.035,.90,'Same six models and tuning grid; all regulatory partners retained.\nHigher values mean worse change prediction than predicting no change.',va='top',fontsize=8)
handles,labs=axes[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',bbox_to_anchor=(.55,.12),ncol=1,frameon=False,fontsize=7)
fig.text(.035,.08,'Pooled CPM/hour squared error, three forecast intervals and both cell-half directions.\nBars: 95% chromosome-bootstrap ranges, not biological-replicate uncertainty.\nSelected fitting/tuning is a post-primary exploratory sensitivity; both populations change.',va='top',fontsize=6.5)
for ext,kw in [('png',{'dpi':600}),('pdf',{}),('svg',{})]:fig.savefig(FIG/('02_selected_fit_comparison_170mm.'+ext),**kw)
plt.close(fig)
pd.DataFrame(rows).to_csv(FIG/'selected_fit_comparison_plot_data.csv',index=False)
print(pd.DataFrame(rows).query('model=="flexible_cis"').to_string(index=False))
