"""Preparation attrition, separated from query planner outcomes."""
import json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path('outputs/results/transfer_scene_evaluation_v1')


def main():
    names=list(json.loads((ROOT/'design.json').read_text())['scenes']);data=[json.loads((ROOT/'libraries'/n/'preparation.json').read_text()) for n in names];grasps=['x110','x115','x120','x125'];matrix=np.zeros((8,4),dtype=int);labels=[]
    stage_map={'pickup_ik':1,'delivery_ik':1,'whole_body_gcs':2,'carrying_iris_cover':2,'sampled_validation':3}
    for i,d in enumerate(data):
        rr=[]
        for j,g in enumerate(grasps):
            a=next(x for x in d['attempts'] if x['name']==g);matrix[i,j]=0 if a['status']=='accepted' else stage_map.get(a.get('stage'),4);rr.append('pass' if a['status']=='accepted' else {1:'IK',2:'region',3:'audit',4:'other'}[matrix[i,j]])
        labels.append(rr)
    from matplotlib.colors import ListedColormap,BoundaryNorm
    colors=['#91c8ae','#e7b66f','#b3a4ce','#e69898','#bbbbbb'];fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.25,1]});ax.imshow(matrix,cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,5),5),aspect='auto')
    ax.set(xticks=np.arange(4),xticklabels=grasps,yticks=np.arange(8),yticklabels=names,title='Shared candidate preparation (32 attempts)')
    for i in range(8):
        for j in range(4):ax.text(j,i,labels[i][j],ha='center',va='center',fontsize=10)
    total=np.array([d['goal_count'] for d in data]);connected=np.array([sum(x['region_graph_connected'] for x in d['connectivity']) for d in data]);y=np.arange(8)
    bx.barh(y,total,color='#dddddd',label='Valid action pairs');bx.barh(y,connected,color='#5793be',label='Carry graph connected');bx.invert_yaxis();bx.set(yticks=y,yticklabels=names,xlim=(0,30),xlabel='Finite grasp / place pairs',title='Action availability vs. region connectivity')
    for i,(a,b) in enumerate(zip(connected,total)):bx.text(b+.25,i,f'{a}/{b}',va='center',fontsize=9)
    bx.legend(loc='lower right',fontsize=8);fig.tight_layout();fig.savefig(ROOT/'preparation_matrix.png',dpi=170);fig.savefig(ROOT/'preparation_matrix.svg');plt.close(fig)
    summary={'candidate_attempts':32,'accepted_candidates':int(np.sum(matrix==0)),'rejected_stage_counts':dict(Counter(a.get('stage','unknown') for d in data for a in d['attempts'] if a['status']!='accepted')),'goal_attempts':sum(len(d['goal_attempts']) for d in data),'valid_goal_pairs':int(total.sum()),'carry_connected_pairs':int(connected.sum()),'not_connected':[{'scene':name,**x} for name,d in zip(names,data) for x in d['connectivity'] if not x['region_graph_connected']]};(ROOT/'preparation_analysis.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='not_connected'}))


if __name__=='__main__':main()
