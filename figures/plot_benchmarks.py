"""Render the benchmark figure from the three packaged result records."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
records = ROOT / "numeral-evidence" / "records"

def read(name):
    return json.loads((records / name).read_text(encoding="utf8"))

def render(vector=None, page_number=None):
    bench = read("bench-summary.json")["systems"]
    m1 = read("m1-report-values.json")["headconf_compare"]
    m5 = read("m5-report-values.json")["policies"]
    navy, blue, teal, rust, grey = "#152e45", "#3c6aaf", "#007e78", "#bc5430", "#5d6d7c"
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.spines.left":False,"axes.edgecolor":"#acb6bc",
                         "axes.axisbelow":True,"text.color":navy,"xtick.color":grey,"ytick.color":navy,
                         "pdf.fonttype":42})
    fig = plt.figure(figsize=(595.3/72,841.9/72))
    def caption(x,y,s,size=10,bold=False,color=navy):
        fig.text(x,y,s,va="top",fontsize=size,color=color,weight="bold" if bold else "normal",linespacing=1.45)
    def bars(rect,values,labels,colors,limit,ticks,fmt,errors=None):
        ax = fig.add_axes(rect)
        ax.barh(range(len(values)), values, height=.55, color=colors)
        if errors:
            ax.errorbar(values,range(len(values)),xerr=errors,fmt="none",ecolor=navy,capsize=3,lw=1.1)
        for i,value in enumerate(values):
            x = value + (errors[1][i] if errors else 0)
            ax.text(x+limit*.012,i,fmt(value),va="center",fontsize=8.6,weight="bold",clip_on=False)
        ax.set_xlim(0,limit);ax.set_ylim(len(values)-.45,-.55);ax.set_yticks(range(len(values)),labels)
        ax.set_xticks(ticks);ax.grid(axis="x",color="#e1e6eb",lw=.6);ax.tick_params(axis="y",length=0,pad=7)
        ax.tick_params(axis="x",length=3,labelsize=8)
        return ax
    caption(.08,.955,"Hindi speech QA benchmarks",22,True)
    caption(.08,.912,"Yash Kumar Sharma   /   GramVaani development experiments",10,False,grey)
    caption(.08,.855,"1   Koyal and SraVaani preserve the same number of numerals",12,True)
    ids=["koyal","sravaani","qwen","whisper"]; labels=["Koyal Hindi 120M","SraVaani 1.0","Qwen3-ASR 1.7B","Whisper large-v3"]
    for k,(left,title) in enumerate([(.275,"Word error rate (%)"),(.655,"Correct numerals (of 110)")]):
        caption(left,.811,title,9,True)
        vals=[bench[i]["wer_published" if k==0 else "correct"] for i in ids]
        bars([left,.628,.232,.148],vals,labels if k==0 else [""]*4,[blue]*4 if k==0 else [teal]*4,
             60 if k==0 else 110,[0,30,60] if k==0 else [0,55,110],(lambda v:f"{v:.1f}") if k==0 else str)
    caption(.08,.587,"On clean speech, Koyal made fewer word errors than SraVaani. Both preserved 94 of\n110 reference numerals. The four systems were evaluated on the same 200 recordings.",9)
    caption(.08,.506,"2   Encoder disagreement ranks numeral errors more accurately",12,True)
    signals=[m1["auroc"][k] for k in ["head_confidence","encoder_margin"]]; vs=[d["auroc"] for d in signals]
    errors=[[v-d["ci"][0] for v,d in zip(vs,signals)],[d["ci"][1]-v for v,d in zip(vs,signals)]]
    a=bars([.31,.39,.54,.084],vs,["Decoder confidence","Encoder disagreement"],[blue,teal],1,[0,.25,.5,.75,1],lambda v:f"{v:.3f}",errors)
    caption(.08,.346,"AUROC on 348 scored slots from 243 sources, including 22 errors. Whiskers show 95%\nconfidence intervals. The paired gain was +0.428 (95% CI +0.312 to +0.537).",9)
    caption(.08,.267,"3   Routing changed which numerals were checked, but yielded no QA gain",12,True)
    keys=["CASCADE","ENC_b25","ALWAYS_qwen","ORACLE_qwen"]
    bars([.325,.105,.52,.128],[m5[k]["correct"] for k in keys],["Baseline cascade","Encoder router (25 listens)","Always substitute","Oracle (reference aided)"],[blue,teal,rust,grey],26,[0,5,10,15,20,26],lambda v:f"{v}/26")
    caption(.08,.062,"On 26 questions, always substituting repaired one answer and broke three. Only one\nquestion contained a corrupted numeral in its evidence span.",9)
    if page_number: caption(.925,.035,str(page_number),8,color=grey)
    fig.savefig(ROOT/"figures"/"hindi-speech-benchmarks.png",dpi=180,facecolor="white")
    if vector: fig.savefig(vector,format="pdf",metadata={"Author":"Yash Kumar Sharma"})
    plt.close(fig)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--vector",type=Path)
    parser.add_argument("--page-number",type=int)
    args=parser.parse_args()
    render(args.vector,args.page_number)
