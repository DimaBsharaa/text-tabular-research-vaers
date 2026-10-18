"""Build notebook 08 with runnable local/Colab refinement, diagnostics and plots."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[3]
cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(text).strip() + '\n'})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': textwrap.dedent(text).strip() + '\n', 'outputs': [], 'execution_count': None})


md('''
    # 08 — Repair v9's care boundary and test it on matched rows

    v10 preserves observed symptoms before later care, excludes explicit post-event background,
    and keeps all structured predictors and labels intact. It is retrospective report classification.
    The policy was fixed before these CPU scores. **These are development TF-IDF diagnostics, not E5/TAR.**

    Local outputs below come from the actual supplied data. Open this notebook from GitHub in a fresh
    Colab CPU runtime and Run all; it clones our code automatically. No tools ZIP upload is needed.
    Use a separate CPU runtime while GPU notebooks continue. Raw yearly ZIPs and v9 remain on Drive.

    **Pipeline:** raw 2019/2023/2024 VAERS ZIPs → notebook 06 / `build_vaers.py` → frozen build-8 v9
    → this notebook / `refine_v9.py` (verified raw sources + frozen v9) → build-9 v10 and controls
    → notebook 02 on GPU. Since frozen v9 already exists, skip 06 for the current run.
''')
code('''
    from pathlib import Path
    import base64, os, sys, json, hashlib, subprocess
    REPO_URL = 'https://github.com/DimaBsharaa/text-tabular-research-vaers.git'

    def github_env():
        # The repo is private: git authenticates with the GH_TOKEN Colab secret. The token travels in an
        # HTTP header set through environment variables, so it never appears in a printed command,
        # an error message or .git/config.
        try:
            from google.colab import userdata
            token = userdata.get('GH_TOKEN')
        except Exception:
            token = None
        if not token:
            raise RuntimeError('GH_TOKEN missing: add a GitHub token with read access to '
                               'DimaBsharaa/text-tabular-research-vaers in Colab Secrets')
        header = base64.b64encode(f'x-access-token:{token}'.encode()).decode()
        return {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_CONFIG_COUNT': '1',
                'GIT_CONFIG_KEY_0': 'http.https://github.com/.extraheader',
                'GIT_CONFIG_VALUE_0': f'AUTHORIZATION: basic {header}'}
    try:
        from google.colab import drive
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    if IN_COLAB:
        drive.mount('/content/drive')
        PROJECT = Path('/content/text-tabular-research-vaers')
        if not PROJECT.exists():
            subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(PROJECT)], check=True, env=github_env())
        else:
            subprocess.run(['git', '-C', str(PROJECT), 'pull', '--ff-only'], check=True, env=github_env())
        print('project commit:', subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip())
        HOME_DATA = Path('/content/drive/MyDrive/text-tabular-research-2')
        BASE = HOME_DATA / 'candidates/build8/vaers_v9/vaers_v9.csv'
        RAW = sorted((HOME_DATA / 'vaers_raw').glob('*VAERSData.zip'))
        BUILD_DIR = HOME_DATA / 'candidates/build9'
        RESULT_ZIP = HOME_DATA / 'benchmark-results/20b24baf903ae7ac/tennis_diagnostic_20b24baf903ae7ac_results.zip'
        DIAGNOSTIC_DIR = HOME_DATA / 'diagnostics/v10_care_boundary2'
    else:
        PROJECT = Path.cwd()
        if not (PROJECT / 'research/scripts/vaers/refine_v9.py').exists():
            PROJECT = PROJECT / 'text-tabular-research-vaers'
        BASE = Path.home() / 'Downloads/vaers_v9-20261017T080754Z-1-001.zip'
        RAW = [Path.home() / f'Downloads/{year}VAERSData.zip' for year in [2019, 2023, 2024]]
        BUILD_DIR = PROJECT.parent / 'vaers-research-cache/derived/build9'
        RESULT_ZIP = Path.home() / 'Downloads/tennis_diagnostic_20b24baf903ae7ac_results.zip'
        DIAGNOSTIC_DIR = PROJECT.parent / 'vaers-research-cache/diagnostics/v10_care_boundary2'
    sys.path.insert(0, str(PROJECT))
    from research.scripts.vaers import refine_v9 as rv, diagnose_refinement as diagnostic
    print('policy:', rv.POLICY_VERSION, '| source sha256:', hashlib.sha256(Path(rv.__file__).read_bytes()).hexdigest())
    print('parent:', BASE)
''')
code('''
    test = subprocess.run([sys.executable, '-m', 'unittest', 'research.scripts.vaers.test_refine_v9', '-q'],
                          cwd=PROJECT, capture_output=True, text=True)
    print(test.stderr)
    if test.returncode:
        raise RuntimeError('Refinement regression tests failed')
    if (BUILD_DIR / 'build_summary.json').exists():
        build = json.loads((BUILD_DIR / 'build_summary.json').read_text())
        paths = {v: str(BUILD_DIR / f'vaers_{v}/vaers_{v}.csv') for v in ['v10', 'v10_matched', 'v9_control']}
        for v, path in paths.items():
            s = json.loads(Path(path).with_name('schema.json').read_text())
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == s['csv_sha256']
            assert s['provenance']['refinement_sha256'] == hashlib.sha256(Path(rv.__file__).read_bytes()).hexdigest(), 'Policy changed; select a new build folder'
        counts = build['counts']
        print('Verified existing frozen build; no files overwritten')
    else:
        paths, counts = rv.build(BASE, RAW, BUILD_DIR)
    import pandas as pd
    from IPython.display import display
    display(pd.DataFrame(counts['by_label']).rename(index={'parent':'Parent rows', 'retained':'Retained rows', 'removed':'Excluded rows'}))
    print('Total:', counts['parent_rows'], '->', counts['kept_rows'], 'rows')
    print('Changed excerpts:', counts['excerpt_changed'], '| changed backgrounds:', counts['background_changed'])
''')
md('''
    ## Actual text changes

    `v10_matched` keeps all parent IDs for sensitivity analysis, including censored placeholders.
    `v10` excludes uninformative excerpts and explicitly post-event background.
    `v9_control` retains the original v9 text on exactly the v10 rows: this is the matched GPU control.
''')
code('''
    parent, _, _ = rv.read_base(BASE)
    parent = parent.set_index('report_id')
    fixed = pd.read_csv(paths['v10_matched'], dtype={'report_id': str}).set_index('report_id')
    eligible = set(pd.read_csv(paths['v10'], dtype={'report_id': str}).report_id)
    examples = []
    for rid in ['809322', '820772', '2554658', '2760881', '831877', '854775']:
        if rid in parent.index:
            examples.append({'ID': rid, 'v9 excerpt': parent.at[rid, 'narrative_onset'],
                             'v10 excerpt': fixed.at[rid, 'narrative_onset'], 'eligible': rid in eligible})
    with pd.option_context('display.max_colwidth', 180):
        display(pd.DataFrame(examples))
''')
md('''
    ## Matched CPU diagnostics — 32 small fits

    The all-parent comparison isolates text changes while allowing censoring placeholders.
    The eligible-row comparison fits both policies on identical restricted training/test rows.
    All fitted preprocessing uses training rows only. These are previously inspected development folds.
''')
code('''
    if (DIAGNOSTIC_DIR / 'summary.json').exists():
        result = json.loads((DIAGNOSTIC_DIR / 'summary.json').read_text())
        for v, path in paths.items():
            assert result['plan']['data_sha256'][v] == hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert hashlib.sha256((DIAGNOSTIC_DIR / 'diagnostic_source.py').read_bytes()).hexdigest() == result['plan']['diagnostic_code_sha256']
        print('Loaded the recorded CPU run; data and saved execution source verified')
    else:
        result = diagnostic.run(BASE, BUILD_DIR, RESULT_ZIP, DIAGNOSTIC_DIR)
    scores = pd.DataFrame(result['scores'])
    means = scores[scores.protocol != 'negative_control'].groupby(['protocol', 'variant', 'condition']).auroc.mean().unstack()
    for protocol in means.index.get_level_values(0).unique():
        means.loc[(protocol, 'v10'), 'structured'] = means.loc[(protocol, 'v9'), 'structured']
    means['joint_signal_proxy'] = means.joint - means[['structured', 'text']].max(axis=1)
    display(means.round(6))
    display(pd.DataFrame(result['quality_checks'].items(), columns=['Implemented check', 'Result']))
    display(pd.DataFrame(result['selection_flags']).T)
''')
code('''
    import matplotlib.pyplot as plt
    common = means.loc['same_eligible_rows', ['structured', 'text', 'joint']].reindex(['v9', 'v10'])
    ax = common.T.plot.bar(figsize=(8, 4), ylim=(0.5, 1.0), rot=0, color=['#718096', '#168a74'])
    ax.set_ylabel('Mean development AUROC')
    ax.set_title('Same eligible rows: TF-IDF/logistic CPU proxy')
    ax.legend(['Original v9 text', 'Corrected v10 text'])
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', fontsize=9)
    plt.tight_layout()
    plt.show()
    display(scores[scores.protocol == 'negative_control'][['variant', 'auroc', 'n_train', 'n_test']])
''')
md('''
    ## Next GPU tests, in two stages

    First run `v10_matched` in notebook 02 with LightGBM/CatBoost and fold 0: **8 evaluations**.
    Compare with the completed v9 fold-0 results on exactly the same parent IDs; no baseline rerun is needed.
    This is a sensitivity test, not the final cohort, because censored placeholders remain.

    Then, if worth extending, run `v9_control` and `v10` on fold 0: **16 evaluations total**.
    They have identical eligible IDs and labels, measuring the correction on the actual candidate cohort.
    Keep the existing benchmark pin and tokens. No E5/TAR improvement has yet been measured.
''')
code('''
    print('MODE = "diagnostic"')
    print('MODELS_OVERRIDE = ["light", "cat"]')
    print('FOLDS_OVERRIDE = [0]')
    for variant in ['v10_matched', 'v9_control', 'v10']:
        folder = f'/content/drive/MyDrive/text-tabular-research-2/candidates/build9/vaers_{variant}'
        print(f'\\n{variant}:\\nCSV_PATH = "{folder}/vaers_{variant}.csv"\\nSCHEMA_PATH = "{folder}/schema.json"')
    print('No official benchmark source was changed. Results here are CPU proxies only.')
''')

nb = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python'}, 'colab': {'name': '08_vaers_v10_refinement.ipynb'}}}
path = ROOT / 'research/notebooks/08_vaers_v10_refinement.ipynb'
path.write_text(json.dumps(nb, indent=1) + '\n')
print('Wrote', path)
