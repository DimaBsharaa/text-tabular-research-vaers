"""Execute ordinary Python notebook cells locally and save standard Jupyter outputs.

This lightweight runner uses the current Python process, not a separate Jupyter kernel.
It supports this notebook's Python cells, display(), stdout, and matplotlib show().
Open/run the resulting notebook in normal Jupyter to use its usual kernel execution.
"""
from pathlib import Path
import base64
import contextlib
import html
import io
import json
import os
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/text-tabular-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import IPython.display

def render_markdown(source):
    """Small static renderer for this notebook's headings, lists, emphasis and links."""
    def inline(text):
        text = html.escape(text)
        text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<a href="\2">\1</a>',text)
        text = re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
        text = re.sub(r'`([^`]+)`',r'<code>\1</code>',text)
        return text
    out = []
    for block in source.strip().split('\n\n'):
        if block.startswith('#'):
            heading, _, title = block.partition(' ')
            level = min(len(heading),6)
            out.append(f'<h{level}>{inline(title)}</h{level}>')
        elif all(line.startswith('- ') for line in block.splitlines()):
            out.append('<ul>'+''.join(f'<li>{inline(line[2:])}</li>' for line in block.splitlines())+'</ul>')
        else:out.append('<p>'+inline(block).replace('\n','<br>')+'</p>')
    return '\n'.join(out)

def main():
    notebook_path = ROOT / 'notebooks/01_multabench_benchmark_analysis.ipynb'
    notebook = json.loads(notebook_path.read_text())
    original_dir = Path.cwd()
    os.chdir(ROOT)
    namespace = {'__name__':'__main__'}
    original_display, original_show = IPython.display.display, plt.show
    execution_count = 0
    started = time.monotonic()
    try:
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':continue
            execution_count += 1
            cell['execution_count'] = execution_count
            outputs = []
            stream = io.StringIO()
            def flush():
                value = stream.getvalue()
                if value:outputs.append({'output_type':'stream','name':'stdout','text':value})
                stream.seek(0);stream.truncate(0)
            def display(*objects, **kwargs):
                flush()
                for obj in objects:
                    data = {'text/plain':str(obj)}
                    if hasattr(obj,'_repr_html_'):
                        representation = obj._repr_html_()
                        if representation:data['text/html']=representation
                    outputs.append({'output_type':'display_data','metadata':{},'data':data})
            def show(*args, **kwargs):
                flush()
                for number in plt.get_fignums():
                    figure = plt.figure(number)
                    image = io.BytesIO()
                    figure.savefig(image,format='png',dpi=110,bbox_inches='tight')
                    outputs.append({'output_type':'display_data','metadata':{},'data':{
                        'image/png':base64.b64encode(image.getvalue()).decode(),
                        'text/plain':f'Matplotlib figure {number}'}})
                plt.close('all')
            IPython.display.display, plt.show = display, show
            namespace['display'] = display
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(compile(cell['source'],f"{notebook_path.name}:cell{execution_count}",'exec'),namespace)
            flush()
            cell['outputs'] = outputs
            print(f'Executed cell {execution_count}: {len(outputs)} outputs',flush=True)
    finally:
        IPython.display.display, plt.show = original_display, original_show
        os.chdir(original_dir)
    notebook['metadata']['research_execution'] = {'runner':'scripts/execute_notebook.py; in-process ordinary Python cells',
        'python':sys.version,'network_access':False,'duration_seconds':round(time.monotonic()-started,2)}
    notebook_path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False))
    blocks = []
    for cell in notebook['cells']:
        if cell['cell_type']=='markdown':blocks.append(render_markdown(cell['source']));continue
        blocks.append('<details><summary>Python cell '+str(cell['execution_count'])+'</summary><pre>'+html.escape(cell['source'])+'</pre></details>')
        for output in cell['outputs']:
            if output['output_type']=='stream':blocks.append('<pre class="stream">'+html.escape(output['text'])+'</pre>')
            elif 'image/png' in output.get('data',{}):blocks.append('<img alt="Executed notebook chart" src="data:image/png;base64,'+output['data']['image/png']+'">')
            elif 'text/html' in output.get('data',{}):blocks.append('<div class="table">'+output['data']['text/html']+'</div>')
            else:blocks.append('<pre>'+html.escape(output['data'].get('text/plain',''))+'</pre>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MulTaBench exploratory notebook</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 24px;color:#213746;background:#fafbfc}h1,h2{line-height:1.25;color:#17465e}h2{margin-top:48px;border-top:1px solid #ccd6de;padding-top:24px}pre{white-space:pre-wrap;font-size:12px;background:#eff3f6;padding:16px;overflow:auto}details{margin:18px 0;color:#547086}.table{overflow:auto;margin:18px 0}table{border-collapse:collapse;font-size:12px;line-height:1.4}td,th{padding:7px 9px;border-bottom:1px solid #d6e0e5;text-align:right}tbody tr:nth-child(even){background:#edf3f7}img{max-width:100%;height:auto;background:white;margin:20px 0}a{color:#176f97}code{background:#e7eef2;padding:1px 4px}</style><main>'+ '\n'.join(blocks)+'</main></html>'
    html_path = notebook_path.with_suffix('.html')
    html_path.write_text(page)
    figures = sum('image/png' in output.get('data',{}) for cell in notebook['cells'] for output in cell.get('outputs',[]))
    assert figures >= 13
    assert all(c.get('execution_count') for c in notebook['cells'] if c['cell_type']=='code')
    assert not any(o['output_type']=='error' for c in notebook['cells'] for o in c.get('outputs',[]))
    print(f'Saved {execution_count} executed cells, {figures} figures, and standalone HTML in {time.monotonic()-started:.1f}s.')

if __name__=='__main__':main()
