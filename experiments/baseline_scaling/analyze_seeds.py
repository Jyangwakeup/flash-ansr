"""Joint descriptive analysis of seeds 0, 1, 2; raw results remain unchanged."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/baseline_scaling'
OUT=BASE/'combined'; OUT.mkdir(exist_ok=True)
seeds=[0,1,2]; grid=[50,100,300,500,700,1000,3000]
rows=[]
for seed in seeds:
    a=[json.loads(l) for l in (BASE/f'seed{seed}'/'baseline.jsonl').read_text().splitlines() if l.strip()]
    assert len(a)==840
    assert {(r['problem'],r['draws'],r['seed']) for r in a}=={(p,n,seed) for p in range(120) for n in grid}
    rows.extend(a)
lookup={(r['seed'],r['problem'],r['draws']):r for r in rows}
for p in range(120):
    assert len({(lookup[s,p,n]['data_seed'],lookup[s,p,n]['expression'],lookup[s,p,n]['model']) for s in seeds for n in grid})==1
counts={key:np.array([[sum(lookup[s,p,n][key] for p in range(120)) for n in grid] for s in seeds]) for key in ['solved_oracle','solved_top1']}
summary={'n_problems':120,'seeds':seeds,'n_rows':len(rows),'errors':sum('error' in r for r in rows),'uncertainty':'sample SD across three seed-level rates; not a confidence interval','per_N':{}}
for j,n in enumerate(grid):
    e={}
    for key,a in counts.items():
        rates=a[:,j]/120*100
        e[key]={'counts':a[:,j].tolist(),'mean_percent':float(rates.mean()),'sample_sd_pp':float(rates.std(ddof=1)),'min_percent':float(rates.min()),'max_percent':float(rates.max())}
    e['mean_model_gflops']=float(np.mean([r['flops'] for r in rows if r['draws']==n])/1e9)
    e['median_fit_seconds_by_seed']=[float(np.median([lookup[s,p,n]['fit_time'] for p in range(120)])) for s in seeds]
    summary['per_N'][n]=e
hits=np.array([[sum(lookup[s,p,n]['solved_oracle'] for s in seeds) for n in grid] for p in range(120)])
summary['oracle_seed_hit_distribution']={n:{str(k):int(sum(hits[:,j]==k)) for k in range(4)} for j,n in enumerate(grid)}
summary['never_solved_any_budget_any_seed']=int(sum(np.all(hits==0,axis=1)))
summary['adjacent_oracle_changes']={}
for a,b in zip(grid,grid[1:]):
    summary['adjacent_oracle_changes'][f'{a}->{b}']=[{'seed':s,'gained':sum(not lookup[s,p,a]['solved_oracle'] and lookup[s,p,b]['solved_oracle'] for p in range(120)),'lost':sum(lookup[s,p,a]['solved_oracle'] and not lookup[s,p,b]['solved_oracle'] for p in range(120))} for s in seeds]
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
colors=['#2563eb','#ea580c','#16a34a']
for ax,key,title in [(axs[0,0],'solved_oracle','A. Oracle: any fitted candidate'),(axs[0,1],'solved_top1','B. Top-1: returned answer')]:
    rates=counts[key]/120*100
    for i,s in enumerate(seeds):ax.plot(grid,rates[i],'.--',color=colors[i],alpha=.7,label=f'Seed {s}')
    ax.errorbar(grid,rates.mean(axis=0),yerr=rates.std(axis=0,ddof=1),fmt='o-',color='#172554',lw=2,capsize=4,label='Mean ± 1 sample SD')
    ax.set(xscale='log',ylim=(15,65),xlabel='Draw budget N',ylabel='Solved problems (%)',title=title);ax.set_xticks(grid,labels=grid);ax.legend(fontsize=9)
ax=axs[1,0];x=np.arange(len(grid));bottom=np.zeros(len(grid))
for k,c,label in [(0,'#cbd5e1','0/3 seeds solved'),(1,'#fbbf24','1/3 seeds solved'),(2,'#38bdf8','2/3 seeds solved'),(3,'#2563eb','3/3 seeds solved')]:
    v=np.sum(hits==k,axis=0);ax.bar(x,v,bottom=bottom,color=c,label=label);bottom+=v
ax.set_xticks(x,labels=grid);ax.set(title='C. Oracle consistency on the same 120 problems',xlabel='Draw budget N',ylabel='Number of problems',ylim=(0,145));ax.legend(fontsize=8,ncol=2)
ax=axs[1,1]
for key,c,label in [('solved_oracle','#2563eb','Oracle'),('solved_top1','#ea580c','Top-1')]:
    rates=counts[key]/120*100;xs=[summary['per_N'][n]['mean_model_gflops'] for n in grid]
    ax.plot(xs,rates.mean(axis=0),'o-',color=c,label=label)
for j,n in enumerate(grid):ax.annotate(str(n),(xs[j],counts['solved_oracle'][:,j].mean()/1.2),xytext=(0,14 if n==700 else 5),textcoords='offset points',ha='center',fontsize=9)
ax.set(xscale='log',ylim=(15,65),title='D. Mean solve rate vs estimated model compute',xlabel='Mean encoder + decoder GFLOPs / problem',ylabel='Mean solved (%)');ax.legend()
for ax in axs.flat:ax.grid(axis='y',alpha=.2)
fig.suptitle('Flash-ANSR baseline | 120 fastsrb problems | seeds 0, 1, 2',fontsize=18)
for ext in ['png','svg']:fig.savefig(OUT/f'baseline_three_seeds.{ext}',dpi=180)
plt.close(fig)
lines=['# 三 seed 独立采样基线联合分析','','## Material Passport','','- Origin Skill: experiment-agent; Origin Mode: validate','- Origin Date: 2026-11-18','- Verification Status: ANALYZED（重新汇总，未重跑模型）','','## 数据与口径','','三个 seed 各 840 条，合计 2520 条，覆盖相同 120 题与七档预算；唯一键、题目表达式、模型与 data_seed 一致性校验通过。error 条数：'+str(summary['errors'])+'。','成功定义仍为留出点 FVU ≤ float32 eps，不等于符号等价。均值为三个 seed 成功率的算术平均；± 为 seed 间样本标准差（ddof=1），不是置信区间。重复使用同一批 120 题，不能把每档 360 条记录当成 360 个独立题目。','','## 全部结果','','| N | Oracle 命中题数 seed0 / 1 / 2 | Oracle 均值 ± SD | Top-1 均值 ± SD |','|---|---|---|---|']
for n,e in summary['per_N'].items():
    o=e['solved_oracle'];t=e['solved_top1'];lines.append(f'| {n} | {" / ".join(map(str,o["counts"]))} | {o["mean_percent"]:.2f}% ± {o["sample_sd_pp"]:.2f} pp | {t["mean_percent"]:.2f}% ± {t["sample_sd_pp"]:.2f} pp |')
lines+=['','## 核心结论','','- N=1000 不再只引用 seed0 的 46.7%，联合结果应使用上表三 seed 的均值及波动。','- N=500 的 seed0 回落没有在另外两个 seed 上复现：300→500 的命中数变化分别为 −3、+4、+5。单 seed 的局部下降不能代表预算增大的平均效果；三 seed 均值随七档预算递增。','- 500 与 700 是同一实验的普通预算点，统一展示。','- Oracle 与 Top-1 在多个点存在小差距；排序损失较小，但不能再说两者处处完全相同。','- N=1000→3000 的 Oracle 命中数变化分别为 +4、+7、+4；收益在三个 seed 中均为正。只有三个 seed，不能据此声称已饱和或证明普遍稳定。',f'- 所有七档、三个 seed 均未命中的题目为 {summary["never_solved_any_budget_any_seed"]} 道。“至少一个 seed 命中”仅用于诊断，不等于一次运行的成功率。','','## 同题稳定性','','| N | 0/3 seed 命中 | 1/3 | 2/3 | 3/3 |','|---|---|---|---|---|']
for n,d in summary['oracle_seed_hit_distribution'].items():lines.append(f'| {n} | '+ ' | '.join(str(d[str(k)]) for k in range(4))+' |')
lines+=['','## 成本与限制','','FLOPs 沿用脚本的编码器和解码器估算，不含完整常数拟合成本。原始耗时包含已知异常长记录，且运行条件未控制，因此联合主图使用估算模型计算量，不以墙钟耗时推断性能优势。','三个 seed 提供初步随机波动信息，样本标准差仅作描述；不对总体成功率作独立二项置信区间，也未对多个预算差异作显著性宣称。数据集代表性及题目间依赖仍有限。','','## 后续精修对照','','使用相同题目、data_seed、fit seed 和误差标准做逐题配对比较；同时记录初始采样与精修的总成本。100、300、500、700 已有三 seed 参照，按实际总计算量选择独立采样对照。','','## Fallacy Scan','','11/11 checked：Simpson（相同题目组成）；生态谬误（汇总不代表每题）；选择偏差（不外推全部公式）；collider（无结果筛选）；基率（120题分母）；均值回归（不依据单 seed 回落定论）；幸存者（全组合）；多重检验（仅描述，无显著性主张）；分析自由度（原始数据保留，SD口径明确）；因果夸大（未声称精修有效）；反向因果（预算预设）。','重复题目不作为独立样本；跨 seed 描述一致性不等于模型重跑可重复性验证。','','## 产物','','- baseline_three_seeds.png / .svg：三 seed 曲线、均值与波动、逐题稳定性、成本曲线。','- summary.json：各 seed 计数、均值、标准差、逐题稳定性与相邻预算变化。']
(OUT/'analysis.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
