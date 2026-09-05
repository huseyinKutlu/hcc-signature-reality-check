import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size':9,'font.family':'sans-serif','axes.linewidth':0.8,
                     'xtick.direction':'out','ytick.direction':'out','savefig.bbox':'tight'})
OUT='figures'; os.makedirs(OUT,exist_ok=True)
ARCH=['resenc','umamba','swin']
LBL={'resenc':'nnU-Net ResEnc-M\n(CNN)','umamba':'U-Mamba Bot\n(SSM)','swin':'SwinUNETR\n(Transformer)'}
SHORT={'resenc':'ResEnc-M','umamba':'U-Mamba','swin':'SwinUNETR'}
C={'resenc':'#2c6fbb','umamba':'#1b9e77','swin':'#d95f02'}

def save(fig,name):
    fig.savefig(f'{OUT}/{name}.png',dpi=300); fig.savefig(f'{OUT}/{name}.pdf')
    plt.close(fig); print(f'  -> {OUT}/{name}.png / .pdf')

def boot_ci(x,n=2000,seed=0):
    r=np.random.default_rng(seed); x=np.asarray(x)
    m=r.choice(x,(n,len(x)),replace=True).mean(1)
    return np.percentile(m,2.5),np.percentile(m,97.5)

df=pd.read_csv('percase_combined3.csv')
sel=pd.read_csv('selective_input.csv')
INT={'resenc':[.6964,.7136,.5703,.7229,.7243],
     'umamba':[.6383,.6671,.5992,.7253,.6821],
     'swin':[.5115,.5191,.4352,.5845,.3439]}

# ---- Fig 1: internal vs external ----
print('Figure 1...')
fig,ax=plt.subplots(figsize=(6.5,3.4))
groups=['Internal\n(5-fold CV)','3D-IRCADb\n(n=15)','LiTS\n(n=118)']
w=0.26
for i,a in enumerate(ARCH):
    vals,los,his=[],[],[]
    v=np.array(INT[a]); vals.append(v.mean()); los.append(v.std(ddof=1)); his.append(v.std(ddof=1))
    for coh in ['ircadb','lits']:
        s=df[df.cohort==coh][a].values; m=s.mean(); lo,hi=boot_ci(s)
        vals.append(m); los.append(m-lo); his.append(hi-m)
    x=np.arange(3)+(i-1)*w
    ax.bar(x,vals,w,color=C[a],label=SHORT[a],edgecolor='white',linewidth=.6)
    ax.errorbar(x,vals,yerr=[los,his],fmt='none',ecolor='#333',elinewidth=.9,capsize=2.5)
    for xi,vi in zip(x,vals): ax.text(xi,vi+0.012,f'{vi:.3f}',ha='center',fontsize=6.5)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(groups)
ax.set_ylabel('Dice similarity coefficient'); ax.set_ylim(0,0.85)
ax.axvline(0.5,color='#bbb',ls='--',lw=.8)
ax.legend(frameon=False,ncol=3,loc='upper right',fontsize=8)
ax.text(0,-0.28,'Error bars: SD across folds (internal); bootstrap 95% CI (external)',
        transform=ax.transAxes,fontsize=6.5,color='#555')
ax.spines[['top','right']].set_visible(False)
save(fig,'Fig1_internal_vs_external')

# ---- Fig 2: case-level concordance ----
print('Figure 2...')
pairs=[('resenc','umamba'),('resenc','swin'),('umamba','swin')]
fig,axes=plt.subplots(1,3,figsize=(7.2,2.7),sharex=True,sharey=True)
for ax,(a,b) in zip(axes,pairs):
    for coh,mk,al in [('lits','o',.55),('ircadb','^',.9)]:
        s=df[df.cohort==coh]
        ax.scatter(s[a],s[b],s=16,marker=mk,alpha=al,edgecolors='none',
                   c='#2c6fbb' if coh=='lits' else '#d95f02')
    from scipy.stats import spearmanr
    rl,_=spearmanr(df[df.cohort=='lits'][a],  df[df.cohort=='lits'][b])
    ri,_=spearmanr(df[df.cohort=='ircadb'][a],df[df.cohort=='ircadb'][b])
    ax.plot([0,1],[0,1],ls='--',c='#999',lw=.8)
    ax.set_xlabel(f'{SHORT[a]} Dice'); ax.set_ylabel(f'{SHORT[b]} Dice')
    ax.set_title(f'LiTS $\\rho$={rl:.3f}   IRCADb $\\rho$={ri:.3f}',fontsize=8)
    ax.set_xlim(-.03,1); ax.set_ylim(-.03,1)
    ax.spines[['top','right']].set_visible(False)
axes[0].legend(handles=[Line2D([],[],marker='o',ls='',color='#2c6fbb',label='LiTS'),
                        Line2D([],[],marker='^',ls='',color='#d95f02',label='3D-IRCADb')],
               frameon=False,fontsize=7,loc='upper left')
save(fig,'Fig2_case_concordance')

# ---- Fig 3: volume stratification ----
print('Figure 3...')
order=['<5 mL','5-20 mL','20-50 mL','>50 mL']
d3=df.copy(); d3['vol_grp']=pd.Categorical(d3.vol_grp,order,ordered=True)
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.2,3.0))
w=0.26
for i,a in enumerate(ARCH):
    g=d3.groupby('vol_grp',observed=True)[a]
    a1.bar(np.arange(4)+(i-1)*w,g.mean(),w,color=C[a],label=SHORT[a],edgecolor='white',linewidth=.6)
    a2.plot(np.arange(4),g.apply(lambda s:(s==0).mean())*100,'o-',color=C[a],ms=4,lw=1.4,label=SHORT[a])
ns=d3.groupby('vol_grp',observed=True).size()
a1.set_xticks(np.arange(4)); a1.set_xticklabels([f'{o}\n(n={ns[o]})' for o in order],fontsize=7.5)
a1.set_ylabel('Dice similarity coefficient'); a1.legend(frameon=False,fontsize=7.5)
a2.set_xticks(np.arange(4)); a2.set_xticklabels(order,fontsize=7.5)
a2.set_ylabel('Complete failure rate (%)'); a2.legend(frameon=False,fontsize=7.5)
for ax in (a1,a2): ax.spines[['top','right']].set_visible(False)
a1.set_xlabel('Reference lesion volume'); a2.set_xlabel('Reference lesion volume')
save(fig,'Fig3_volume_stratification')

# ---- Fig 4: reliability ----
print('Figure 4...')
fig,axes=plt.subplots(1,2,figsize=(7.0,3.2),sharey=True)
for ax,coh,ttl in zip(axes,['ircadb','lits'],['3D-IRCADb (n=15)','LiTS (n=118)']):
    ax.plot([0,1],[0,1],ls='--',c='#999',lw=.9,label='Perfect calibration')
    for a in ARCH:
        f=f'reliability_{coh}_{a}.csv'
        if not os.path.exists(f): print(f'   [skip] {f}'); continue
        r=pd.read_csv(f); r=r[r.n>0]
        ax.plot(r.conf,r.acc,'-',color=C[a],lw=1.4,label=SHORT[a])
        w=r.n/r.n.max()
        ax.scatter(r.conf,r.acc,s=6+34*w,color=C[a],edgecolors='none',zorder=3)
    ax.set_title(ttl,fontsize=9); ax.set_xlabel('Predicted probability')
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Observed frequency'); axes[0].legend(frameon=False,fontsize=7,loc='lower right')
axes[1].text(.97,.06,'marker size $\\propto$ voxels per bin',transform=axes[1].transAxes,
             fontsize=6.5,color='#555',ha='right')
for ax in axes:
    ax.text(.22,.72,'under-confident',transform=ax.transAxes,fontsize=6.5,color='#888',
            style='italic',rotation=32)
    ax.text(.72,.52,'over-confident',transform=ax.transAxes,fontsize=6.5,color='#888',
            style='italic',rotation=32,ha='center')
save(fig,'Fig4_reliability')

# ---- Fig 5: risk-coverage ----
print('Figure 5...')
def rc(sub,score):
    s=sub.sort_values(score,ascending=False).reset_index(drop=True)
    cov=np.arange(0.3,1.001,0.05); out=[]
    for c in cov:
        k=max(1,int(round(len(s)*c)))
        out.append((c,s.dice[:k].mean(),(s.dice[:k]==0).mean()*100))
    return np.array(out)
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.2,3.0))
L=sel[sel.cohort=='lits'].copy(); L['mean_conf_pred']=L.mean_conf_pred.fillna(0)
for a in ARCH:
    s=L[L.arch==a]
    c1=rc(s,'mean_conf_pred'); c2=rc(s,'pred_vol_ml')
    a1.plot(c1[:,0]*100,c1[:,1],'-',color=C[a],lw=1.6,label=f'{SHORT[a]} (confidence)')
    a1.plot(c2[:,0]*100,c2[:,1],':',color=C[a],lw=1.3)
    a2.plot(c1[:,0]*100,c1[:,2],'-',color=C[a],lw=1.6,label=SHORT[a])
a1.set_xlabel('Coverage (% of cases retained)'); a1.set_ylabel('Dice on retained cases')
a2.set_xlabel('Coverage (% of cases retained)'); a2.set_ylabel('Complete failure rate (%)')
a1.legend(frameon=False,fontsize=7)
a1.text(.02,.02,'solid: confidence   dotted: predicted volume',transform=a1.transAxes,fontsize=6.5,color='#555')
for ax in (a1,a2): ax.spines[['top','right']].set_visible(False); ax.invert_xaxis()
save(fig,'Fig5_risk_coverage')
print('\nTamamlandi.')
