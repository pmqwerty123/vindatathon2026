import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
FIG=Path('figures'); FIG.mkdir(exist_ok=True)
OUT=Path('outputs/submissions'); DATA=Path('data/raw')
fig, ax = plt.subplots(figsize=(13,5.8)); ax.axis('off')
boxes=[('Raw daily sales\nDate, Revenue, COGS',.04,.62),('Calendar/event features\nFourier, holidays, Tet, promos',.25,.62),('Log-target learners\nLightGBM, Ridge, HW fallback',.48,.62),('Quarter specialists\nQ-specific LightGBM weighting',.48,.25),('Blend + scale\n0.10 stat + 0.10 ridge + 0.80 LGB',.70,.62),('COGS margin rule\nodd/even quarterly ratios',.70,.25),('Final forecasts\nRevenue + COGS submission',.88,.45)]
for text,x,y in boxes:
    ax.text(x,y,text,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=0.45',fc='#F4F7FB',ec='#294C60',lw=1.5))
for a,b in [((.13,.62),(.18,.62)),((.34,.62),(.40,.62)),((.57,.62),(.62,.62)),((.57,.31),(.63,.50)),((.76,.57),(.83,.48)),((.77,.31),(.83,.43)),((.48,.52),(.48,.37))]:
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.6,color='#294C60'))
ax.set_title('Final Forecasting Pipeline',fontsize=16,weight='bold',pad=15); fig.tight_layout(); fig.savefig(FIG/'model_pipeline.png',dpi=220,bbox_inches='tight'); plt.close(fig)
sales=pd.read_csv(DATA/'sales.csv',parse_dates=['Date']); sales['Y']=sales.Date.dt.year; sales['Q']=sales.Date.dt.quarter
res=pd.read_csv(OUT/'oof_rolling_experiment_results.csv'); best=res[res.candidate=='best_grid']; pivot=best.pivot_table(index='alpha',columns='cr',values='combined_mae',aggfunc='mean')
fig,ax=plt.subplots(figsize=(7.5,5.2)); im=ax.imshow(pivot.values,cmap='YlGnBu_r',aspect='auto'); ax.set_xticks(np.arange(len(pivot.columns))); ax.set_xticklabels([f'{c:.2f}' for c in pivot.columns]); ax.set_yticks(np.arange(len(pivot.index))); ax.set_yticklabels([f'{i:.2f}' for i in pivot.index])
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val=pivot.values[i,j]; ax.text(j,i,f'{val/1e6:.2f}M',ha='center',va='center',fontsize=10)
ax.set_xlabel('Revenue scale CR'); ax.set_ylabel('Specialist blend ALPHA'); ax.set_title('Calibration Search: Mean Best-Fold MAE Surface',weight='bold'); fig.colorbar(im,ax=ax,label='Combined MAE'); fig.tight_layout(); fig.savefig(FIG/'calibration_search.png',dpi=220,bbox_inches='tight'); plt.close(fig)
q=sales.groupby(['Y','Q']).agg({'Revenue':'sum','COGS':'sum'}).reset_index(); q['ratio']=q.COGS/q.Revenue; q['x']=pd.to_datetime(q['Y'].astype(str) + '-' + ((q['Q']-1)*3+1).astype(str) + '-01')
fig,ax=plt.subplots(figsize=(11,4.8))
for quarter,g in q.groupby('Q'):
    ax.plot(g.x,g.ratio,marker='o',lw=1.5,label=f'Q{quarter}')
ax.axhline(1.0,color='firebrick',ls='--',lw=1,label='COGS = Revenue'); ax.set_ylabel('COGS / Revenue'); ax.set_xlabel('Quarter'); ax.set_title('Quarterly COGS-to-Revenue Ratio',weight='bold'); ax.legend(ncol=5,fontsize=9); ax.grid(alpha=.25); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG/'cogs_ratio.png',dpi=220,bbox_inches='tight'); plt.close(fig)
v=res.pivot(index='fold',columns='candidate',values='combined_mae').reset_index(); fig,ax=plt.subplots(figsize=(8.5,4.8)); ax.plot(v.fold,v.v57_default,marker='o',lw=2,label='Selected calibration'); ax.plot(v.fold,v.best_grid,marker='o',lw=2,label='Best grid calibration'); ax.set_xticks(v.fold); ax.set_ylabel('Combined MAE'); ax.set_xlabel('Validation year'); ax.set_title('Rolling-Origin Validation Curve',weight='bold'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'validation_curve.png',dpi=220,bbox_inches='tight'); plt.close(fig)
sub=pd.read_csv(OUT/'submission.csv',parse_dates=['Date']); fig,ax=plt.subplots(figsize=(12,5)); ax.plot(sub.Date,sub.Revenue/1e6,lw=1.2,label='Revenue'); ax.plot(sub.Date,sub.COGS/1e6,lw=1.2,label='COGS'); ax.set_ylabel('Forecast value (millions)'); ax.set_xlabel('Date'); ax.set_title('Final Forecast Trajectory',weight='bold'); ax.grid(alpha=.25); ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG/'final_forecast.png',dpi=220,bbox_inches='tight'); plt.close(fig)
print('done')
