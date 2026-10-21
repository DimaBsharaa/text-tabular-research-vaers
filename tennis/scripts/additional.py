"""Derived coverage and bounded publisher spot checks, without publishing text."""
import html
import io
import json
import re

import pandas as pd

from tennis.scripts.common import CACHE, OUT, PRIVATE, normalized, write_json


def main():
    a=pd.read_csv(PRIVATE/"variant_A.csv");c=pd.read_csv(PRIVATE/"variant_C.csv")
    source=pd.read_csv(PRIVATE/"interviews_with_answers.csv")
    extra={"a_text_age_days":{}, "corpus_top10_player_interview_share":float(source.player.value_counts().head(10).sum()/len(source)),
        "corpus_grandslam_interview_share":float((source.tournament_type=="Grand Slam").mean()),
        "corpus_source_winner_interview_share":float(source.result.mean()),
        "source_word_count_over_512_fraction":float((source.answer_words>512).mean()),
        "a_gender_counts":a.gender.value_counts().to_dict(),
        "b_eligible_completed_male_matches_after_snapshot":len(c[(c.gender=="M")&(c.match_date>"2015-06-11")])}
    for side in ["a","b"]:
        ages=(pd.to_datetime(a.match_date)-pd.to_datetime(a[f"{side}_text_date"])).dt.days
        extra["a_text_age_days"][side]={"median":float(ages.median()),"p90":float(ages.quantile(.9)),"max":int(ages.max()),
                                    "max_doc_reuse":int(a[f"{side}_interview_id"].value_counts().max())}
    links=pd.read_csv(OUT/"cornell_match_join_audit.csv",dtype={"interview_id":str});links=links[links.join_status=="matched"]
    matches=pd.read_csv(PRIVATE/"matches.csv");by_player={};records=[]
    for side in ["a","b"]:
        for (g,k),group in matches.groupby(["gender",f"player_{side}_key"]):
            by_player.setdefault((g,k),[]).extend(group[["match_id","match_date","comment"]].to_dict("records"))
    for row in links.itertuples():
        options=[v for v in by_player[(row.gender,row.player_key)] if v["match_date"]>row.date]
        if not options:continue
        day=min(v["match_date"] for v in options);picks=[v for v in options if v["match_date"]==day]
        if len(picks)==1 and (pd.Timestamp(day)-pd.Timestamp(row.date)).days<=90 and picks[0]["comment"]=="Completed":
            records.append({"interview_id":row.interview_id,"match_id":picks[0]["match_id"],"date":day,"source_date":row.date,"gender":row.gender})
    secondary=pd.DataFrame(records);secondary.to_csv(OUT/"playercentric_next_match_links.csv",index=False)
    extra.update(playercentric_source_documents_with_next_match=len(secondary),playercentric_unique_target_matches=int(secondary.match_id.nunique()),playercentric_publication_verified=0)
    write_json(OUT/"additional_audit.json",extra)
    original=json.loads((CACHE/"cornell/transcripts_matchinfo.json").read_text());checks=[]
    for file,iid,url in [
        ("asap_murray_20150621.html","0","https://www.asapsports.com/show_interview.php?id=110501"),
        ("asap_110504.html","0","https://www.asapsports.com/show_interview.php?id=110504"),
        ("asap_murray_20150822.html","497","https://www.asapsports.com/show_interview.php?id=112348")]:
        path=CACHE/"sources"/file
        if not path.exists():continue
        body=html.unescape(re.sub("<[^>]+>"," ",path.read_text(encoding="utf8",errors="replace")))
        row=original[iid];prefix=" ".join(row["QandA"][0][1].split()[:20])
        checks.append({"interview_id":iid,"publisher_url":url,"player":row["player"],"date":row["date"],
            "first_answer_prefix_matches_publisher":normalized(prefix) in normalized(body),
            "historical_publish_timestamp_verified":False,
            "note":"Same-day semifinal is a deliberate negative control" if "110501" in url else "Dated publisher-content corroboration; not a historical publication audit"})
    write_json(OUT/"publisher_spot_checks.json",checks)
    point_bytes=(CACHE/"sources/mcp_points_prefix.csv").read_bytes()
    points=pd.read_csv(io.BytesIO(point_bytes[:point_bytes.rfind(b"\n")+1]),encoding="latin1")
    notes=points.Notes.dropna().astype(str)
    write_json(OUT/"mcp_point_prefix_audit.json",{"scope":"bounded first 65536 bytes, final partial CSV line dropped; not representative",
        "rows":len(points),"columns":points.columns.tolist(),"unique_matches":int(points.match_id.nunique()),
        "nonempty_notes":len(notes),"notes_at_least_5_words":int((notes.str.split().str.len()>=5).sum()),
        "shot_code_examples":{col:points[col].dropna().astype(str).head(3).tolist() for col in ["1st","2nd"]}})
    print(json.dumps(extra,indent=2));print(json.dumps(checks,indent=2))


if __name__ == "__main__":main()
