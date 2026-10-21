"""Training-review extension: select whole Q&A blocks to preserve context.

Added after inspecting sentence-extraction failures on training interviews. It is
versioned separately from the original five-policy freeze and never uses targets
for passage selection. The reserved test remains unscored.
"""
import gzip
import json
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from tennis.scripts.a_text import normalize, topic_flags, WEIGHTS, body_tokens, cap_head, count_tokens
from tennis.scripts.a_study import STUDY, model_fingerprint
from tennis.scripts.common import CACHE, OUT, ROOT, digest, write_json
from tennis.scripts.prototypes import NUMERIC, CONTEXT
from tennis.scripts.diagnostics import preprocessor


def select_blocks(qa):
    candidates = []
    for idx, (question, answer) in enumerate(qa):
        question, answer = normalize(question), normalize(answer)
        if not answer:
            continue
        flags = topic_flags(question + " " + answer)
        score = sum(WEIGHTS[k] for k, v in flags.items() if v)
        include_question = len(answer.split()) < 80
        rendered = f"Q: {question} A: {answer}" if include_question else answer
        candidates.append({"pair_idx": idx, "text": rendered, "score": score,
            "tokens": body_tokens(rendered), "topics": [k for k, v in flags.items() if v],
            "question_included": include_question, "answer_sha256": digest(answer), "question_sha256": digest(question)})
    budget = 512-count_tokens(""); selected = []; covered = set(); used = 0
    pool = [v for v in candidates if v["score"] > 0]
    while pool:
        pool.sort(key=lambda v: (-(v["score"] + 2*len(set(v["topics"])-covered))/max(1,v["tokens"])**.25, v["pair_idx"]))
        item = pool.pop(0)
        if used+item["tokens"] <= budget:
            selected.append(item); used += item["tokens"]; covered.update(item["topics"])
    if selected:
        selected.sort(key=lambda v: v["pair_idx"])
        text = cap_head(" ".join(v["text"] for v in selected))
        lineage = [{k: v[k] for k in ["pair_idx", "question_included", "answer_sha256", "question_sha256"]} for v in selected]
        fallback = False
    else:
        source = next((v["text"] for v in sorted(candidates, key=lambda v: (-v["score"], v["pair_idx"]))), "")
        text = cap_head(source); lineage = []; fallback = True
    return {"text": text, "lineage": lineage, "selected_sha256": digest(text),
        "selected_tokens_a": count_tokens(text), "selected_tokens_b": count_tokens(text,"player_b_text"),
        "fallback_clipped_whole_block": fallback}


def main():
    config = {"policy": "qa_blocks", "rule": "fixed topical whole-answer blocks; question retained for answers under 80 words",
        "reason": "Training-only qualitative review found pronoun/opponent ambiguity and figurative injury cues in sentence extraction",
        "selector_source_sha256": digest((ROOT/'tennis/scripts/a_context_blocks.py').read_bytes()),
        "uses_target_labels_for_selection": False, "reserved_test_scored": False,
        "original_five_policy_freeze_unchanged": True, "default_selected_policy_unchanged": "topical_answers"}
    config["config_sha256"] = digest(json.dumps(config,sort_keys=True))
    path = STUDY/'context_blocks_frozen_before_diagnostics.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Context-block probe changed; explicitly version it before rerunning')
    write_json(path,config)
    original = json.loads((CACHE/'cornell/transcripts_matchinfo.json').read_text())
    frame = pd.read_csv(STUDY/'metadata.csv')
    ids = sorted(set(frame.a_interview_id.astype(int))|set(frame.b_interview_id.astype(int)))
    cache = {iid: select_blocks(original[str(iid)]['QandA']) for iid in ids}
    with gzip.open(STUDY/'qa_blocks_private.jsonl.gz','wt') as stream:
        for iid, value in cache.items(): stream.write(json.dumps({'interview_id':iid}|value)+'\n')
    exported = pd.read_csv(STUDY/'features_topical_answers.csv')
    for side in ['a','b']:
        exported[f'player_{side}_text'] = frame[f'{side}_interview_id'].map(lambda iid: cache[int(iid)]['text'])
    exported.to_csv(STUDY/'features_qa_blocks.csv',index=False)
    schema=json.loads((STUDY/'schema.json').read_text())
    schema.update(selected_policy='qa_blocks',csv_sha256=digest((STUDY/'features_qa_blocks.csv').read_bytes()))
    write_json(STUDY/'schema_qa_blocks.json',schema)
    indexed=exported.set_index('match_id');tr=frame[frame.split=='train'];dv=frame[frame.split=='development']
    train=indexed.loc[tr.match_id].copy();dev=indexed.loc[dv.match_id].copy()
    for d in [train,dev]:
        for c in CONTEXT:
            if c!='best_of':d[c]=d[c].fillna('missing').astype(str)
    scores=[];fit=[]
    for condition in ['text_tfidf','joint_tfidf']:
        structured=[] if condition=='text_tfidf' else NUMERIC+CONTEXT
        text=['player_a_text','player_b_text']
        model=make_pipeline(preprocessor(structured,NUMERIC+['best_of'],text),
                            LogisticRegression(C=1,solver='liblinear',max_iter=500,random_state=42))
        started=time.perf_counter();model.fit(train[structured+text],train.target)
        evidence,before=model_fingerprint(model)
        pred=model.predict_proba(dev[structured+text])[:,1]
        pd.DataFrame({'match_id':dev.index,'target':dev.target,'probability':pred}).to_csv(OUT/f'a_v1_dev_qa_blocks_{condition}_predictions.csv',index=False)
        scores.append({'policy':'qa_blocks','condition':condition,'auroc':float(roc_auc_score(dev.target,pred)),
            'train_rows':len(train),'development_rows':len(dev),'reserved_test_rows_scored':0,
            'runtime_seconds':time.perf_counter()-started,'official_result':False,'config_sha256':config['config_sha256']})
        fit.append({'condition':condition,'fit_row_ids':train.index.tolist(),'development_row_ids':dev.index.tolist(),
            'preprocessor_sha_before':before,'preprocessor_sha_after':model_fingerprint(model)[1],
            'reserved_test_rows_used_for_fit_or_score':0})
        print('Context-block diagnostic',condition,round(scores[-1]['auroc'],4),flush=True)
    pd.DataFrame(scores).to_csv(OUT/'a_v1_context_blocks_development_scores.csv',index=False)
    write_json(OUT/'a_v1_context_blocks_fit_audit.json',fit)
    train_ids=set(tr.a_interview_id.astype(int))|set(tr.b_interview_id.astype(int))
    metrics=[]
    for iid,value in cache.items():
        source=' '.join(normalize(v[1]) for v in original[str(iid)]['QandA'])
        item={'interview_id':iid,'in_training_documents':iid in train_ids,'policy':'qa_blocks',
              'formatted_tokens_a':value['selected_tokens_a'],'formatted_tokens_b':value['selected_tokens_b'],
              'selected_sha256':value['selected_sha256']}
        for k in WEIGHTS:
            item[k+'_full']=topic_flags(source)[k];item[k+'_selected']=topic_flags(value['text'])[k]
        metrics.append(item)
    pd.DataFrame(metrics).to_csv(OUT/'a_v1_context_blocks_metrics.csv',index=False)
    write_json(OUT/'a_v1_context_blocks_policy.json',config)


if __name__=='__main__':main()
