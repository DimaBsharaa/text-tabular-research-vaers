"""Fetch bounded research sources into an external cache; no model installation.

ConvoKit archive 71.2 MB, original archive 15.9 MB; parsed 580 MB SpaCy payload is
never extracted by audit.py. Excel reader is a 96.6 KB pure-Python wheel, imported
as a zip rather than installed into the user's Python environment.
"""
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from research.scripts.tennis.common import CACHE, OUT, SNAPSHOT, write_json


def fetch(url, target, limit):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        request = urllib.request.Request(url, headers={"User-Agent":"tennis-research/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            expected = int(response.headers.get("Content-Length", 0))
            if expected > limit: raise ValueError(f"Download exceeds declared cap: {url}")
            temp = target.with_suffix(target.suffix + ".part")
            total = 0
            with temp.open("wb") as out:
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > limit: raise ValueError(f"Download exceeded cap: {url}")
                    out.write(block)
            temp.replace(target)
    return {"url":url, "cache_path":str(target.relative_to(CACHE)), "bytes":target.stat().st_size,
            "sha256":hashlib.sha256(target.read_bytes()).hexdigest()}


def main():
    entries = []
    for url, name, limit in [
        ("https://zissou.infosci.cornell.edu/convokit/datasets/tennis-corpus/tennis-corpus.zip", "tennis-corpus.zip", 90000000),
        ("https://www.cs.cornell.edu/~liye/tennis_data.zip", "tennis_data.zip", 20000000)]:
        entries.append(fetch(url, CACHE / "cornell" / name, limit))
    index = CACHE / "sources/tennis-data-index.html"
    entries.append(fetch("https://www.tennis-data.co.uk/alldata.php", index, 200000))
    links = re.findall(r'href\s*=\s*["\x27]?([^"\x27\s>]+)', index.read_text(encoding="latin1"), re.I)
    for year in range(2007,2017):
        for gender in (["m"] if year == 2016 else ["m","w"]):
            ext = "xlsx" if year >= 2013 else "xls"
            suffix = f"/{year}{'w' if gender == 'w' else ''}/{year}.{ext}"
            candidates = [v for v in links if v.endswith(suffix)]
            if len(candidates) != 1: raise ValueError(f"Cannot uniquely locate historical file {suffix}")
            url = urllib.parse.urljoin("https://www.tennis-data.co.uk/", candidates[0])
            entries.append(fetch(url, CACHE / "sources" / f"{'atp' if gender == 'm' else 'wta'}_{year}.{ext}",2000000))
    for file in ["charting-m-stats-ServeDirection.csv","charting-m-stats-Overview.csv","charting-m-matches.csv","README.md"]:
        entries.append(fetch(f"https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/{SNAPSHOT}/{file}",
                             CACHE / "sources" / ("mcp_2015june_" + file),6000000))
    url = "https://files.pythonhosted.org/packages/1a/62/c8d562e7766786ba6587d09c5a8ba9f718ed3fa8af7f4553e8f91c36f302/xlrd-2.0.2-py2.py3-none-any.whl"
    reader = fetch(url, CACHE / "sources/xlrd-2.0.2.whl",200000)
    assert reader["sha256"] == "ea762c3d29f4cca48d82df517b6d89fbce4db3107f9d78713e48cd321d5c9aa9"
    entries.append(reader)
    prefix=CACHE/"sources/mcp_points_prefix.csv"
    point_url=f"https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/{SNAPSHOT}/charting-m-points.csv"
    if not prefix.exists():
        request=urllib.request.Request(point_url,headers={"Range":"bytes=0-65535"})
        with urllib.request.urlopen(request,timeout=60) as response:
            if response.status!=206 or not response.headers.get("Content-Range","").startswith("bytes 0-65535/"):
                raise ValueError("Point-file range not honored; refusing the full download")
            data=response.read(65537)
            if len(data)!=65536:raise ValueError("Unexpected point-prefix size")
            prefix.write_bytes(data)
    entries.append({"url":point_url,"cache_path":str(prefix.relative_to(CACHE)),"bytes":prefix.stat().st_size,
                    "sha256":hashlib.sha256(prefix.read_bytes()).hexdigest(),"range":"bytes=0-65535"})
    # Preserve the provenance of the actual date dictionary and bounded webpage checks.
    for url,name in [
        ("https://www.tennis-data.co.uk/notes.txt","tennis-data-notes.txt"),
        ("https://www.tennis-data.co.uk/disclaimer.php","tennis-data-disclaimer.html"),
        ("https://api.github.com/repos/JeffSackmann/tennis_MatchChartingProject/commits?path=charting-m-stats-ServeDirection.csv&until=2015-07-01T00:00:00Z&per_page=1","mcp_before2015july.json"),
        ("https://www.asapsports.com/show_player.php?id=15247","asap_murray_index.html"),
        ("https://www.asapsports.com/show_interview.php?id=110501","asap_murray_20150621.html"),
        ("https://www.asapsports.com/show_interview.php?id=110503","asap_110503.html"),
        ("https://www.asapsports.com/show_interview.php?id=110504","asap_110504.html"),
        ("https://www.asapsports.com/show_interview.php?id=112348","asap_murray_20150822.html")]:
        entries.append(fetch(url,CACHE/"sources"/name,500000))
    manifest=OUT/"source_manifest.json"
    if manifest.exists():
        previous={r["cache_path"]:r["sha256"] for r in json.loads(manifest.read_text())}
        for entry in entries:
            if entry["cache_path"] in previous and entry["sha256"] != previous[entry["cache_path"]]:
                raise ValueError(f"Source snapshot drift: {entry['cache_path']}; review before replacing the recorded manifest")
    write_json(OUT / "source_manifest.json", entries)
    print(f"Verified {len(entries)} cached source files; {sum(e['bytes'] for e in entries)/1e6:.1f} MB compressed/source bytes")


if __name__ == "__main__": main()
