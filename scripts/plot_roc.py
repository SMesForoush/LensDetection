import sys, glob, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R, OUT = sys.argv[1], sys.argv[2]
models = [("lensnet","Baseline CNN","#2a78d6","-"),("deeplens","CMU DeepLens","#eb6834","--"),("efficientnet","EfficientNet-B0","#1baf7a",":")]
plt.rcParams.update({"font.size":9,"axes.edgecolor":"#8a8a85","axes.linewidth":0.6,"xtick.color":"#555","ytick.color":"#555"})
fig, axes = plt.subplots(1,2,figsize=(6.3,3.1),sharey=True)
for ax,(fn,title) in zip(axes,[("roc_curve.csv","(a) Synthetic test (PRISM)"),("real_roc_curve.csv","(b) Real set (COSMOS-Web)")]):
    ax.plot([0,1],[0,1],color="#b5b4ad",lw=0.8,zorder=0)
    for m,lab,c,ls in models:
        first=True
        for f in sorted(glob.glob(f"{R}/{m}/seed_*/{fn}")):
            d=pd.read_csv(f); ax.plot(d.fpr,d.tpr,color=c,ls=ls,lw=1.4,label=lab if first else None); first=False
    ax.set_title(title,fontsize=9,loc="left"); ax.set_xlabel("False-positive rate")
    ax.set_xlim(0,1); ax.set_ylim(0,1.005); ax.set_aspect("equal")
    ax.grid(color="#e6e5e0",lw=0.5); ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Recall (true-positive rate)")
axes[1].legend(loc="lower right",frameon=False,fontsize=8,handlelength=2.4)
fig.tight_layout(); fig.savefig(OUT+".pdf"); fig.savefig(OUT+".png",dpi=160)
