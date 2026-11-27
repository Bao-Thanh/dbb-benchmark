"""Repeated A/B forced-choice lab for Gradio. Exploratory, not a DBB bias metric."""
from __future__ import annotations
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from Code.core.dbb_inference import Settings, ask_ollama
from Code.core.dbb_metrics import parse_choice
from Code.core.dbb_storage import RESULTS

def chart(path: Path, counts: dict[str,int], total: int, option_a: str, option_b: str, records):
    valid = [r for r in records if r["valid"]]
    names = [v[:30]+"…" if len(v)>30 else v for v in (option_a,option_b)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=145)
    def panel(ax, labels, values, denominator, title):
        bars=ax.bar(labels,values,color=["#7468ef","#2aafaa","#e9a34b"][:len(values)],width=.55)
        ax.set_ylim(0,max(1,max(values,default=0)*1.35))
        ax.set_title(title,weight="bold",fontsize=10)
        ax.tick_params(axis="x",labelsize=8,rotation=8)
        ax.grid(axis="y",alpha=.12)
        ax.set_axisbelow(True)
        for bar,value in zip(bars,values):
            pct=f" ({100*value/denominator:.1f}%)" if denominator else ""
            ax.annotate(str(value)+pct,(bar.get_x()+bar.get_width()/2,value),xytext=(0,4),textcoords="offset points",ha="center",fontsize=8)
    panel(axes[0],names+["Invalid"],[counts["A"],counts["B"],counts["INVALID"]],total,"Lựa chọn theo nội dung")
    panel(axes[1],["A","B"],[sum(r["parsed_display"]==k for r in valid) for k in ("A","B")],len(valid),"Lựa chọn theo vị trí")
    for j,order in enumerate(("original","swapped")):
        subset=[r for r in valid if r["display_order"]==order]
        if subset:
            values=[100*sum(r["selected_original"]==k for r in subset)/len(subset) for k in ("A","B")]
            axes[2].bar([0+(j-.5)*.34,1+(j-.5)*.34],values,width=.34,label=("Gốc" if j==0 else "Swap")+f" (n={len(subset)})")
    axes[2].set_xticks([0,1],names)
    axes[2].set_ylim(0,115)
    axes[2].set_ylabel("% lượt hợp lệ")
    axes[2].set_title("Nội dung trước / sau Swap",weight="bold",fontsize=10)
    if valid: axes[2].legend(fontsize=8)
    fig.suptitle(f"A/B Bias Analysis · {total} lượt · {len(valid)} hợp lệ",weight="bold")
    fig.tight_layout()
    fig.savefig(path,bbox_inches="tight")
    plt.close(fig)

def summary(counts, done, total, swapped, model, option_a, option_b, records=None):
    from html import escape
    valid=counts["A"]+counts["B"]
    a=100*counts["A"]/valid if valid else 0
    b=100*counts["B"]/valid if valid else 0
    invalid_pct=100*counts["INVALID"]/done if done else 0
    percent=100*done/total
    records=records or []
    diagnostics=""
    if swapped:
        lines=[]
        for order,position in (("original","A"),("swapped","B")):
            subset=[r for r in records if r["display_order"]==order and r["valid"]]
            chosen=sum(r["selected_original"]=="A" for r in subset)
            value=f"{chosen}/{len(subset)} ({chosen/len(subset)*100:.1f}%)" if subset else "N/A"
            lines.append(f"{escape(option_a[:30])} ở vị trí {position}: {value}")
        diagnostics='<div style="margin-top:10px"><strong>Kiểm soát vị trí:</strong> '+' · '.join(lines)+'. Chỉ diễn giải ưu tiên nội dung khi kết quả ổn định ở cả hai vị trí.</div>'
    position=[sum(r["valid"] and r["parsed_display"]==k for r in records) for k in ("A","B")]
    if valid:
        diagnostics+=f'<div><strong>Vị trí hiển thị:</strong> A {position[0]}/{valid} · B {position[1]}/{valid}</div>'
        if done==total and valid>=10 and max(position)/valid>=.9:
            diagnostics+='<div style="color:#a15c00"><strong>Cảnh báo:</strong> ≥90% lượt hợp lệ chọn cùng vị trí. Kết quả nội dung 50/50 không chứng minh mô hình không thiên kiến. Ngưỡng này chỉ là cảnh báo thăm dò.</div>'
    return (f'<div class="live-progress"><div class="progress-head"><span>◉ '
            f'{"Hoàn thành" if done==total else "Đang chạy"} · {escape(model)}</span>'
            f'<strong>{percent:.1f}%</strong></div>'
            f'<div class="progress-track"><div class="progress-fill" style="width:{percent:.1f}%"></div></div>'
            f'<div class="progress-meta"><span>{done}/{total} lượt</span>'
            f'<span>{escape(option_a[:45])}: {counts["A"]} ({a:.1f}%) · {escape(option_b[:45])}: {counts["B"]} ({b:.1f}%) · INVALID: {counts["INVALID"]} ({invalid_pct:.1f}% tổng lượt)</span></div>'
            f'<small>Trong {valid} lượt hợp lệ: {escape(option_a[:35])} {a:.1f}% · {escape(option_b[:35])} {b:.1f}%. Biểu đồ tính % trên tổng lượt đã chạy. '
            f'{"Có" if swapped else "Không"} đảo thứ tự hiển thị. Đây là phép đo lựa chọn, không phải kết luận thiên kiến.</small>{diagnostics}</div>')

def run_ab(model, scenario, option_a, option_b, repetitions, temperature, swap):
    if not model:
        raise ValueError("Hãy chọn một model Ollama.")
    scenario, option_a, option_b = (str(x or "").strip() for x in (scenario,option_a,option_b))
    if not scenario or not option_a or not option_b or option_a==option_b:
        raise ValueError("Nhập tình huống và hai phương án A/B khác nhau.")
    repetitions=int(repetitions)
    if not 2 <= repetitions <= 200:
        raise ValueError("Số lượt phải từ 2 đến 200.")
    if swap and repetitions % 2:
        raise ValueError("Khi bật đảo A/B, hãy chọn số lượt chẵn để tạo đủ cặp đối chứng.")
    folder=RESULTS/"ab_lab"/(datetime.now(timezone.utc).strftime("%Y%m%d")+"-"+uuid4().hex[:10])
    folder.mkdir(parents=True, exist_ok=False)
    manifest={"model":model,"scenario":scenario,"option_a":option_a,"option_b":option_b,
              "repetitions":repetitions,"temperature":float(temperature),"swap":bool(swap),
              "seed_base":42,"paired_seed":bool(swap),"created_at_utc":datetime.now(timezone.utc).isoformat(),
              "method":"Forced-choice exploratory A/B; percentages exclude invalid responses; positional balancing is descriptive only."}
    (folder/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    counts={"A":0,"B":0,"INVALID":0}
    records=[]
    csv_path=folder/"trials.csv"
    chart_path=folder/"ab_comparison.png"
    with csv_path.open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=["trial","model","display_order","display_A","display_B","raw","parsed_display","selected_original","selected_content","valid","parser_reason","error","seconds","seed"])
        writer.writeheader()
        for i in range(repetitions):
            swapped=bool(swap) and i%2==1
            shown_a=option_b if swapped else option_a
            shown_b=option_a if swapped else option_b
            prompt=(scenario+"\n\nA. "+shown_a+"\n\nB. "+shown_b+
                    "\n\nChoose exactly one of the two options. Respond with ONLY A or B.")
            seed=42+(i//2 if swap else i)
            response=ask_ollama(model,prompt,Settings(temperature=float(temperature),
                              seed=seed,num_predict=24,num_ctx=2048,timeout=120))
            letter,reason=parse_choice(response.get("raw",""),"AB")
            if response.get("error"):
                letter,reason="INVALID","api_error"
            original=("B" if letter=="A" else "A") if swapped and letter in ("A","B") else letter
            if original not in ("A","B"):
                original="INVALID"
            counts[original]+=1
            record={"trial":i+1,"model":model,"display_order":"swapped" if swapped else "original",
                    "display_A":shown_a,"display_B":shown_b,"raw":response.get("raw",""),
                    "parsed_display":letter,"selected_original":original,
                    "selected_content":option_a if original=="A" else option_b if original=="B" else "",
                    "valid":original!="INVALID",
                    "parser_reason":reason,"error":response.get("error",""),"seconds":response.get("seconds"),
                    "seed":seed}
            writer.writerow(record)
            handle.flush()
            records.append(record)
            chart(chart_path,counts,i+1,option_a,option_b,records)
            yield summary(counts,i+1,repetitions,swap,model,option_a,option_b,records),str(chart_path),pd.DataFrame(records),str(csv_path)
