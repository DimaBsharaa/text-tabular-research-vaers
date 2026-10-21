"""Secondary text/identity diagnostics; flags are not automatic leakage claims."""
import json
import re
import unicodedata

import numpy as np
import pandas as pd

from tennis.scripts.a_study import STUDY
from tennis.scripts.common import OUT, PRIVATE, write_json
from tennis.scripts.prototypes import NUMERIC


def ascii_text(text):
    return unicodedata.normalize('NFKD',str(text)).encode('ascii','ignore').decode().lower()


def main():
    frame=pd.read_csv(STUDY/'metadata.csv')
    features=pd.read_csv(STUDY/'features_topical_answers.csv').set_index('match_id')
    links=pd.read_csv(OUT/'cornell_match_join_audit.csv')
    good=links[links.join_status=='matched']
    cross=good.groupby(['gender','source_tournament','target_tournament']).size().rename('accepted_interviews').reset_index()
    cross.to_csv(OUT/'a_v1_tournament_name_crosswalk.csv',index=False)
    mappings=cross.groupby(['gender','source_tournament']).target_tournament.nunique()
    ambiguous=mappings[mappings>1]
    flags=[]
    for row in frame.itertuples():
        names=[ascii_text(getattr(row,'player_'+side)).rsplit(' ',1)[0] for side in ['a','b']]
        if min(map(len,names))<4:continue
        for side in ['a','b']:
            text=ascii_text(features.loc[row.match_id,f'player_{side}_text'])
            both=all(re.search(r'\b'+re.escape(name)+r'\b',text) for name in names)
            score=bool(re.search(r'\b[0-7]\s*[-–‑]\s*[0-7]\b',text))
            if both and score:
                flags.append({'match_id':row.match_id,'target_date':row.match_date,'side':side,
                    'interview_id':getattr(row,f'{side}_interview_id'),'source_date':getattr(row,f'{side}_text_date'),
                    'source_match_id':getattr(row,f'{side}_source_match_id'),
                    'source_is_same_match':getattr(row,f'{side}_source_match_id')==row.match_id,
                    'status':'REVIEW: pair names plus score-like string; may describe an earlier encounter'})
    pd.DataFrame(flags,columns=['match_id','target_date','side','interview_id','source_date','source_match_id','source_is_same_match','status']).to_csv(OUT/'a_v1_target_reference_text_flags.csv',index=False)
    label_rows=[]
    for row in frame[frame.split=='train'].itertuples():
        for side in ['a','b']:
            label_rows.append({'interview_id':getattr(row,f'{side}_interview_id'),'text_side':side,'target':row.target})
    labels=pd.DataFrame(label_rows).groupby(['interview_id','text_side']).target.agg(['size','nunique','mean']).reset_index()
    labels.rename(columns={'size':'training_uses','nunique':'distinct_target_labels','mean':'player_a_win_fraction'},inplace=True)
    labels.to_csv(OUT/'a_v1_training_document_label_ambiguity.csv',index=False)
    base=pd.read_csv(PRIVATE/'variant_A.csv').set_index('match_id')
    fields=[]
    for c in NUMERIC:
        fields.append({'feature':c,'rows_changed_from_legacy_history_definition':int((~np.isclose(frame[c].to_numpy(float),base.loc[frame.match_id,c].to_numpy(float),equal_nan=True)).sum())})
    pd.DataFrame(fields).to_csv(OUT/'a_v1_history_definition_changes.csv',index=False)
    summary={'source_tournament_names_with_multiple_target_name_mappings':int(len(ambiguous)),
        'crosswalk_rows':len(cross),'name_difference_is_not_automatic_error':True,
        'target_pair_plus_score_text_flags':len(flags),'any_flag_uses_same_source_target_match':any(v['source_is_same_match'] for v in flags),
        'text_screen_scope':'Exact >=4-character surnames and simple set-score patterns in selected answers; not exhaustive semantic outcome detection',
        'training_interview_side_groups':len(labels),'training_interview_side_groups_with_both_labels':int((labels.distinct_target_labels>1).sum()),
        'interpretation':'Mixed future labels for a reused interview/side are legitimate opponent-dependent supervision, not automatic leakage; they may weaken per-column TAR training.'}
    write_json(OUT/'a_v1_secondary_diagnostics.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
