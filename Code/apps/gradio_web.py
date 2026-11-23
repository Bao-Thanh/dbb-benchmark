"""Gradio DBB Research Studio. Run: python -m Code.apps.gradio_web"""
from __future__ import annotations
import io, os, sys, json, zipfile, subprocess, threading
from pathlib import Path

# Support both VS Code's direct-file Run button and python -m execution.
if __package__ in (None, ''):
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import gradio as gr
import pandas as pd
from Code.core.dbb_data import DATASET, DBB_SHA256, file_sha256, load_dbb, audited_pairs, choose_pairs, load_bbq, load_crows, load_stereoset, sample_rows
from Code.core.dbb_inference import Settings, models_available, BASE
from Code.core.dbb_storage import create_run, folder_for, load_run, export_csv
from Code.core.dbb_status import read_status
from Code.core.dbb_runner import dbb_tasks, bbq_tasks, crows_tasks, stereoset_tasks, evaluate
from Code.core.dbb_metrics import benchmark_summary, latest_attempts
from Code.reporting.dbb_compact_report import sample_disjoint_pairs, METHOD_NOTE, report_files, zip_files

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Results'/'gradio_exports'
OUT.mkdir(parents=True,exist_ok=True)
LOCK=threading.Lock()
PROCS={}
BENCH={'DBB':('Bias-Dataset.csv',load_dbb,dbb_tasks),'BBQ':('BBQ-Dataset.csv',load_bbq,bbq_tasks),'CrowS-Pairs':('CrowS-Pairs.csv',load_crows,crows_tasks),'StereoSet':('StereoSet.csv',load_stereoset,stereoset_tasks)}
CSS="""
.gradio-container{max-width:1480px!important;margin:auto!important;background:#f6f8fc!important;font-family:Inter,system-ui,sans-serif!important}
.hero{padding:52px 58px;border-radius:28px;background:radial-gradient(circle at 90% 0%,#584bb3,transparent 42%),linear-gradient(115deg,#101a33,#283567 75%,#38317a);color:white;box-shadow:0 24px 50px #1b28512b;margin-bottom:24px}
.hero .eyebrow{font-size:12px;letter-spacing:2.4px;color:#c4bcff;font-weight:800}.hero h1{font-size:clamp(35px,5vw,65px)!important;letter-spacing:-2.4px!important;line-height:1.12!important;margin:20px 0!important;color:white!important}.hero h1 span{color:#bdb4ff}.hero p{color:#dbe1fb;font-size:16px;max-width:720px;line-height:1.7}.tags{display:flex;flex-wrap:wrap;gap:10px;margin-top:24px}.tags span{border:1px solid #ffffff36;border-radius:30px;padding:9px 14px;background:#ffffff0e;font-size:12px;color:#f0efff}
.gradio-container .tab-nav{gap:9px;border:0!important;background:transparent!important;margin:20px 0!important}.gradio-container .tab-nav button{border-radius:13px!important;border:1px solid #e2e8f2!important;background:white!important;padding:13px 20px!important;font-weight:700!important;color:#657189!important}.gradio-container .tab-nav button.selected{background:#27315d!important;color:white!important}
.panel{border:1px solid #e2e8f2!important;border-radius:20px!important;background:#fff!important;padding:20px!important;box-shadow:0 10px 30px #182a4a09!important}.gradio-container button.primary{background:linear-gradient(105deg,#7468ef,#5146c7)!important;border:0!important;border-radius:12px!important;font-weight:750!important}.gradio-container button.secondary{border-radius:12px!important}
.stat{border-radius:16px;padding:20px;background:#f4f3ff;border:1px solid #dedafb}.stat small{font-size:12px;color:#6f6b9c;font-weight:750}.stat strong{display:block;font-size:30px;color:#28225f;margin-top:5px}.footer{color:#8290a3;font-size:12px;padding:25px 5px;border-top:1px solid #e0e6f0;margin-top:26px}
@media(max-width:700px){.hero{padding:30px 25px}.hero h1{letter-spacing:-1px!important}}
"""
HERO='<div class="hero"><div class="eyebrow">◈ RESEARCH PLATFORM &nbsp; • &nbsp; LOCAL FIRST</div><h1>Understand the patterns.<br><span>Question the bias.</span></h1><p>DBB Research Studio — phân tích thiên kiến xã hội trong mô hình ngôn ngữ, từ cặp câu hỏi đối chứng đến kết quả có thể kiểm tra và tải về.</p><div class="tags"><span>◈ Counterfactual pairs</span><span>◈ Transparent evaluation</span><span>◈ Local Ollama</span><span>◈ Real results only</span></div></div>'

def export(rid,name,data):
    folder=OUT/rid
    folder.mkdir(parents=True,exist_ok=True)
    p=folder/name
    p.write_bytes(data)
    return str(p)

def model_options():
    models=models_available()
    info='✓ Ollama · '+str(len(models))+' model khả dụng' if models else '⚠ Ollama chưa kết nối; chạy ollama serve và tải model trước.'
    return (gr.update(choices=models,value=models[:1]),gr.update(choices=models,value=models[:1]),gr.update(choices=models,value=models[:1]),gr.update(choices=models,value=models[0] if models else None),info)

def dataset_info():
    p=DATASET/'Bias-Dataset.csv'
    if not p.exists(): return '⚠ Thiếu Dataset/Bias-Dataset.csv — không tạo dữ liệu giả.'
    try: return '✓ Dataset DBB · 77.350 dòng · SHA256 '+('đã xác minh' if file_sha256(p)==DBB_SHA256 else 'KHÔNG KHỚP bản audit')
    except Exception as e: return '⚠ Dataset: '+str(e)

def start_dbb(models,count,category,previous):
    models=list(models or [])
    count=int(count)
    if not models: raise gr.Error('Chọn ít nhất một model Ollama.')
    if count<2 or count%2 or count>2000: raise gr.Error('Số câu hỏi phải chẵn, từ 2 đến 2000.')
    if previous and read_status(previous).get('state') in ('running','initializing'):
        raise gr.Error('Run hiện tại vẫn đang chạy.')
    with LOCK:
        if any(p.poll() is None for p in PROCS.values()): raise gr.Error('Một run đang xử lý. Hãy dừng hoặc chờ.')
        try:
            df=load_dbb()
            pairs=audited_pairs(df)
            chosen=sample_disjoint_pairs(pairs,count,category,42)
            cfg={'workflow':'gradio_compact_v1','models':models,'pair_ids':chosen.pair_id.tolist(),'question_count':count,'group':category,'seed':42,'repeats':10,'settings':Settings(temperature=.8,top_p=1.,seed=42,num_predict=96,num_ctx=2048,timeout=120).metadata(),'dataset_sha256':file_sha256(DATASET/'Bias-Dataset.csv'),'audited_sha256':DBB_SHA256,'method':METHOD_NOTE,'expected_calls_without_retries':count*len(models)*10}
            rid=create_run('experiment1',cfg)
            folder=folder_for('experiment1',rid)
            worker=ROOT/'Code'/'workers'/'dbb_compact_worker.py'
            with (folder/'worker.log').open('a',encoding='utf-8') as log:
                PROCS[rid]=subprocess.Popen([sys.executable,'-u',str(worker),'--run-id',rid],cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            return rid,'✓ Đã khởi chạy '+rid+' · Nhấn Làm mới tiến độ để theo dõi.'
        except Exception as e: raise gr.Error('Không thể bắt đầu: '+str(e))

def stop_dbb(rid):
    if not rid:return 'Chưa có run.'
    folder=folder_for('experiment1',rid)
    if not (folder/'manifest.json').exists():return 'Không tìm thấy run.'
    (folder/'stop.request').touch(exist_ok=True)
    return 'Đã yêu cầu dừng an toàn sau lượt đang xử lý.'

def poll_dbb(rid):
    empty=(None,None,pd.DataFrame(),None)
    if not rid:return ('Chưa có run. Chọn model và bắt đầu.',)+empty
    status=read_status(rid)
    state=status.get('state','initializing')
    proc=PROCS.get(rid)
    if state in ('running','initializing') and proc and proc.poll() not in (None,0):state='failed'
    done,total=int(status.get('done') or 0),int(status.get('total') or 0)
    msg='**Run:** '+rid+' · **Trạng thái:** '+state+' · **Model:** '+str(status.get('model') or '—')+' · **Tiến độ:** '+str(done)+'/'+str(total)+' · **Hợp lệ:** '+str(status.get('valid',0))
    if status.get('error'):msg+='\n\n⚠ '+str(status['error'])[:400]
    if state not in ('completed','stopped','failed'):return (msg,)+empty
    try:
        manifest,attempts=load_run('experiment1',rid)
        cfg=manifest['config']
        if file_sha256(DATASET/'Bias-Dataset.csv')!=cfg['dataset_sha256']:raise ValueError('Dataset thay đổi sau khi chạy.')
        if not attempts:return (msg+' · Không có phản hồi thực.',)+empty
        pairs=audited_pairs(load_dbb())
        pairs=pairs.loc[pairs.pair_id.isin(cfg['pair_ids'])]
        files=report_files(pairs,attempts,cfg['models'],rid,state=='completed',manifest)
        summary=pd.read_csv(io.BytesIO(files['tong_hop.csv']))
        return msg,export(rid,'categories.png',files['bieu_do_nhom.png']),export(rid,'diagnostics.png',files['bieu_do_chi_so.png']),summary,export(rid,'report.zip',zip_files(files))
    except Exception as e:return (msg+'\n\n⚠ Xuất báo cáo thất bại: '+str(e),)+empty

def make_tasks(name,count,cfg):
    filename,loader,builder=BENCH[name]
    if not (DATASET/filename).exists():raise FileNotFoundError('Thiếu Dataset/'+filename)
    df=loader()
    if name=='DBB':return dbb_tasks(df,choose_pairs(audited_pairs(df),count,'Tất cả',42),cfg)
    return builder(sample_rows(df,count,42),cfg)

def compare(models,benches,count,kind,progress=gr.Progress()):
    models=list(models or []);benches=list(benches or []);count=int(count)
    if not models or not benches:raise gr.Error('Chọn model và benchmark.')
    if not 1<=count<=500:raise gr.Error('Số mẫu phải từ 1 đến 500.')
    records=[];errors=[];ids=[]
    total=len(models)*len(benches)
    for i,m in enumerate(models):
        for j,b in enumerate(benches):
            k=i*len(benches)+j
            progress(k/total,desc=m+' · '+b)
            try:
                cfg={'model':m,'settings':Settings(temperature=0.,seed=42,num_predict=96,timeout=120).metadata(),'repeats':1,'workflow':'gradio_comparison_v1','metric_scope':'exploratory_only','benchmarks':benches,'dataset_hashes':{b:file_sha256(DATASET/BENCH[b][0])}}
                tasks=make_tasks(b,count,cfg)
                rid=create_run(kind,cfg);ids.append(rid)
                evaluate(kind,rid,tasks,progress=lambda done,n:progress((k+done/max(n,1))/total,desc=m+' · '+b))
                records+=load_run(kind,rid)[1]
            except Exception as e:errors.append(m+' / '+b+': '+str(e))
    if not records:return '⚠ Không có phản hồi thực. '+'; '.join(errors),pd.DataFrame(),None
    groups=[]
    for m,g in pd.DataFrame(records).groupby('model'):
        frame=benchmark_summary(latest_attempts(g.to_dict('records')))
        if not frame.empty:frame.insert(0,'Model',m);groups.append(frame)
    summary=pd.concat(groups,ignore_index=True) if groups else pd.DataFrame(columns=['Model','Benchmark'])
    planned=pd.MultiIndex.from_product([models,benches],names=['Model','Benchmark']).to_frame(index=False)
    summary=planned.merge(summary,on=['Model','Benchmark'],how='left')
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('attempts.csv',export_csv(records))
        z.writestr('summary.csv',summary.to_csv(index=False).encode('utf-8-sig'))
        z.writestr('method.txt','Exploratory metrics only. DBB single-pass != S; BBQ accuracy != BBQ bias; CrowS/StereoSet are forced-choice proxies.')
        z.writestr('run_ids.txt','\n'.join(ids))
    note='✓ '+str(len(records))+' phản hồi được ghi · '+str(len(ids))+' audit runs. Các chỉ số không so sánh trực tiếp.'
    if errors:note+='\n\n⚠ '+'; '.join(errors[:6])
    return note,summary,export('compare-'+ids[0],'comparison.zip',buffer.getvalue())

def chat(message,history,model,temperature):
    if not model:raise gr.Error('Chọn model Ollama.')
    if not message or not message.strip():return '',history,'Nhập câu hỏi trước.'
    import requests
    history=list(history or [])
    messages=[{'role':'system','content':'You are a helpful assistant. Reply in the user language.'}]
    messages +=[{'role':x['role'],'content':x['content']} for x in history[-12:] if x.get('role') in ('user','assistant') and isinstance(x.get('content'),str)]
    messages.append({'role':'user','content':message})
    try:
        r=requests.post(BASE+'/api/chat',json={'model':model,'stream':False,'think':False,'messages':messages,'options':{'temperature':float(temperature),'num_predict':400}},timeout=180)
        r.raise_for_status()
        answer=str(r.json().get('message',{}).get('content') or '').strip()
        if not answer:raise ValueError('Model trả lời rỗng.')
    except Exception as e:return message,history,'⚠ Ollama: '+str(e)
    from Code.core.dbb_counterfactual import parse_generic_choice
    import re
    indicators=re.findall(r'(?i)\\b(?:all|always|never|tất cả|luôn luôn|bẩm sinh)\\b',answer)
    history +=[{'role':'user','content':message},{'role':'assistant','content':answer}]
    folder=ROOT/'Results'/'explainable'/'gradio_chat'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'audit.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'model':model,'prompt':message,'answer':answer,'heuristic_indicators':len(indicators)},ensure_ascii=False)+'\n')
    return '',history,'**Tín hiệu từ ngữ cần xem xét:** '+str(len(indicators))+' · Heuristic thăm dò, không phải bias score.'

def build():
    with gr.Blocks(title='DBB · Research Studio') as demo:
        gr.HTML(HERO)
        with gr.Row():
            ds=gr.Markdown(dataset_info())
            conn=gr.Markdown('Đang kiểm tra Ollama…')
        with gr.Tabs():
            with gr.Tab('◈ DBB Benchmark'):
                gr.Markdown('## 01 / Counterfactual evaluation\nCùng tập câu hỏi cho mọi model · đảo A/B · 10 lượt/câu · chỉ số DBB S.')
                with gr.Row():
                    with gr.Column(scale=1,elem_classes='panel'):
                        m1=gr.Dropdown(label='Mô hình Ollama',choices=[],multiselect=True)
                        n1=gr.Slider(2,500,value=10,step=2,label='Số câu hỏi (chẵn)')
                        cat=gr.Dropdown(choices=['Tất cả','age','gender','race','religions','socioeconomic_class'],value='Tất cả',label='Nhóm nhân khẩu học')
                        gr.Markdown('Seed 42 · Temperature 0.8 · 5 lượt mỗi vị trí A/B.')
                        start=gr.Button('▶ Bắt đầu đánh giá',variant='primary')
                        stop=gr.Button('■ Dừng an toàn')
                        refresh=gr.Button('↻ Cập nhật tiến độ')
                    with gr.Column(scale=2,elem_classes='panel'):
                        rid=gr.State('')
                        info=gr.Markdown('Chưa chạy. Kết quả chỉ hiển thị khi có dữ liệu thực.')
                        with gr.Row():
                            chart1=gr.Image(label='DBB S ≥ 20 theo nhóm',type='filepath',interactive=False)
                            chart2=gr.Image(label='Chỉ số thăm dò',type='filepath',interactive=False)
                        table1=gr.Dataframe(label='Kết quả theo model và nhóm',interactive=False)
                        file1=gr.File(label='Tải báo cáo ZIP (CSV + PNG + phương pháp)')
                        gr.Markdown('**N/A ≠ 0.** True Social Bias không được coi là một chỉ số DBB chính thức.')
                start.click(start_dbb,[m1,n1,cat,rid],[rid,info]).then(poll_dbb,rid,[info,chart1,chart2,table1,file1])
                stop.click(stop_dbb,rid,info)
                refresh.click(poll_dbb,rid,[info,chart1,chart2,table1,file1])
            with gr.Tab('▦ Four benchmarks'):
                gr.Markdown('## 02 / Benchmark comparison\nDBB · BBQ · CrowS-Pairs · StereoSet. Chỉ số khác định nghĩa, không xếp hạng trực tiếp.')
                with gr.Row():
                    with gr.Column(scale=1,elem_classes='panel'):
                        m2=gr.Dropdown(label='Mô hình',choices=[],multiselect=True)
                        bs=gr.CheckboxGroup(choices=list(BENCH),value=list(BENCH),label='Benchmarks')
                        n2=gr.Slider(1,100,value=10,step=1,label='Số mẫu / benchmark')
                        go2=gr.Button('▶ Chạy so sánh',variant='primary')
                    with gr.Column(scale=2,elem_classes='panel'):
                        info2=gr.Markdown('Chưa có kết quả.')
                        table2=gr.Dataframe(label='Chỉ số mô tả',interactive=False)
                        file2=gr.File(label='Tải CSV + audit ZIP')
                go2.click(lambda m,b,n:compare(m,b,n,'experiment2'),[m2,bs,n2],[info2,table2,file2])
            with gr.Tab('◫ DBB vs BBQ'):
                gr.Markdown('## 03 / DBB vs BBQ\nCùng model, hai tập câu hỏi khác nhau. Không suy luận chúng là các cặp tương đương.')
                with gr.Row():
                    with gr.Column(scale=1,elem_classes='panel'):
                        m3=gr.Dropdown(label='Mô hình',choices=[],multiselect=True)
                        n3=gr.Slider(1,100,value=10,step=1,label='Số mẫu / benchmark')
                        go3=gr.Button('▶ Chạy DBB & BBQ',variant='primary')
                    with gr.Column(scale=2,elem_classes='panel'):
                        info3=gr.Markdown('Chưa có kết quả.')
                        table3=gr.Dataframe(label='Bảng kết quả',interactive=False)
                        file3=gr.File(label='Tải báo cáo ZIP')
                go3.click(lambda m,n:compare(m,['DBB','BBQ'],n,'experiment3'),[m3,n3],[info3,table3,file3])
            with gr.Tab('◇ AI Bias Lab'):
                gr.Markdown('## 04 / Interactive AI Bias Lab\nTrò chuyện trực tiếp với Ollama. Tín hiệu ngôn ngữ chỉ để rà soát, không phải kết luận thiên kiến.')
                with gr.Row():
                    with gr.Column(scale=1,elem_classes='panel'):
                        mc=gr.Dropdown(label='Model chat',choices=[])
                        temp=gr.Slider(0,1,value=.2,step=.05,label='Temperature')
                        gr.Markdown('**Thử nghiệm:** Soạn tình huống hai ứng viên ngang năng lực, sau đó hoán đổi tên/vị trí.')
                        signal=gr.Markdown('Chưa có tín hiệu phân tích.')
                        clear=gr.Button('Xóa cuộc trò chuyện')
                    with gr.Column(scale=2,elem_classes='panel'):
                        history=gr.Chatbot(label='Cuộc trò chuyện',height=460)
                        prompt=gr.Textbox(label='Câu hỏi',lines=3,placeholder='Đặt câu hỏi hoặc mô tả tình huống…')
                        send=gr.Button('Gửi tin nhắn ↗',variant='primary')
                send.click(chat,[prompt,history,mc,temp],[prompt,history,signal])
                prompt.submit(chat,[prompt,history,mc,temp],[prompt,history,signal])
                clear.click(lambda:([],'','Đã xóa giao diện chat.'),outputs=[history,prompt,signal])
        reload=gr.Button('↻ Kết nối / làm mới danh sách model Ollama')
        reload.click(model_options,outputs=[m1,m2,m3,mc,conn])
        demo.load(model_options,outputs=[m1,m2,m3,mc,conn])
        gr.HTML('<div class="footer">DBB RESEARCH STUDIO · Pan et al., Findings EMNLP 2025 · Ollama local · No fabricated results · Experimental indicators are not evidence of discrimination.</div>')
    return demo

if __name__=='__main__':
    app=build()
    app.queue(default_concurrency_limit=2,max_size=24)
    app.launch(server_name=os.getenv('DBB_HOST','127.0.0.1'),server_port=int(os.getenv('DBB_PORT','7860')),show_error=True,
               theme=gr.themes.Soft(primary_hue='violet',neutral_hue='slate'),css=CSS)
