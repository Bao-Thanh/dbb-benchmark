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

def chart(path: Path, counts: dict[str,int], total: int, option_a: str, option_b: str):
    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=145)
    fig.patch.set_facecolor("#ffffff")
    labels = [option_a, option_b, "Không hợp lệ"]
    labels = [(v[:36] + "…" if len(v)>36 else v) for v in labels]
    bars = ax.bar(labels,
                  [counts["A"], counts["B"], counts["INVALID"]],
                  color=["#7468ef", "#2aafaa", "#e9a34b"], width=.56)
    ax.set_ylim(0, max(1, max(counts.values()) * 1.3))
    valid = counts["A"] + counts["B"]
    ax.set_ylabel("Số lượt (trên tổng lượt đã chạy)")
    ax.set_title("Kết quả theo NỘI DUNG phương án · " + str(total) + " lượt", weight="bold", pad=20)
    ax.tick_params(axis="x", labelsize=9, rotation=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.12)
    ax.set_axisbelow(True)
    for bar in bars:
        value = int(bar.get_height())
        pct = f" ({value / total * 100:.1f}%)" if total else ""
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+.06,
                str(value) + pct, ha="center", va="bottom", weight="bold")
    note = (f"Trong {valid} lượt hợp lệ: A {counts['A']/valid*100:.1f}% · B {counts['B']/valid*100:.1f}%" if valid else "Chưa có lượt hợp lệ")
    fig.text(.5, .005, note, ha="center", fontsize=9, color="#657189")
    fig.tight_layout(rect=[0,.04,1,1])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def summary(counts, done, total, swapped, model, option_a, option_b):
    from html import escape
    valid=counts["A"]+counts["B"]
    a=100*counts["A"]/valid if valid else 0
    b=100*counts["B"]/valid if valid else 0
    invalid_pct=100*counts["INVALID"]/done if done else 0
    percent=100*done/total
    return (f'<div class="live-progress"><div class="progress-head"><span>◉ '
            f'{"Hoàn thành" if done==total else "Đang chạy"} · {escape(model)}</span>'
            f'<strong>{percent:.1f}%</strong></div>'
            f'<div class="progress-track"><div class="progress-fill" style="width:{percent:.1f}%"></div></div>'
            f'<div class="progress-meta"><span>{done}/{total} lượt</span>'
            f'<span>{escape(option_a[:45])}: {counts["A"]} ({a:.1f}%) · {escape(option_b[:45])}: {counts["B"]} ({b:.1f}%) · INVALID: {counts["INVALID"]} ({invalid_pct:.1f}% tổng lượt)</span></div>'
            f'<small>Trong {valid} lượt hợp lệ: {escape(option_a[:35])} {a:.1f}% · {escape(option_b[:35])} {b:.1f}%. Biểu đồ tính % trên tổng lượt đã chạy. '
            f'{"Có" if swapped else "Không"} đảo thứ tự hiển thị. Đây là phép đo lựa chọn, không phải kết luận thiên kiến.</small></div>')

def run_ab(model, scenario, option_a, option_b, repetitions, temperature, swap):
    if not model:
        raise ValueError("Hãy chọn một model Ollama.")
    scenario, option_a, option_b = (str(x or "").strip() for x in (scenario,option_a,option_b))
    if not scenario or not option_a or not option_b or option_a==option_b:
        raise ValueError("Nhập tình huống và hai phương án A/B khác nhau.")
    repetitions=int(repetitions)
    if not 2 <= repetitions <= 200:
        raise ValueError("Số lượt phải từ 2 đến 200.")
    folder=RESULTS/"ab_lab"/(datetime.now(timezone.utc).strftime("%Y%m%d")+"-"+uuid4().hex[:10])
    folder.mkdir(parents=True, exist_ok=False)
    manifest={"model":model,"scenario":scenario,"option_a":option_a,"option_b":option_b,
              "repetitions":repetitions,"temperature":float(temperature),"swap":bool(swap),
              "seed_base":42,"created_at_utc":datetime.now(timezone.utc).isoformat(),
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
            seed=42+i
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
            if i==repetitions-1 or i%2==0:
                chart(chart_path,counts,i+1,option_a,option_b)
            yield summary(counts,i+1,repetitions,swap,model,option_a,option_b),str(chart_path),pd.DataFrame(records),str(csv_path)
