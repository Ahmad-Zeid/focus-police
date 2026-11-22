"""Participant study tools: who took part, pseudonymising old recordings, and withdrawing someone.

    python study.py list                       # participants, their recordings and minutes (codes only)
    python study.py pseudonymise NAME CODE     # replace a name used as participant ID with a code, everywhere
    python study.py withdraw CODE              # take a participant's recordings out of the study
    python study.py withdraw CODE --delete     # ... and delete them for good

How the study keeps people apart from their data (see STUDY.md):
  - recordings carry only a random code (P-XXXX) in the participant column and the runner file name;
  - names and contacts, if collected at all, live only in study-private/roster.csv on the experimenter's
    laptop, which git ignores;
  - withdrawal: find the code (the participant has it, or look it up in the roster), run `withdraw`, retrain.
    Without --delete the folders move to sessions/_withdrawn/ (git-ignored, skipped by training) so a
    mistake can be undone; with --delete they are removed. Either way the roster row is marked withdrawn.
"""
import argparse
import csv
import glob
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SESSIONS = os.environ.get("FP_SESSIONS", os.path.join(HERE, "sessions"))
PRIVATE = os.environ.get("FP_PRIVATE", os.path.join(HERE, "study-private"))
ROSTER = os.path.join(PRIVATE, "roster.csv")


def recordings():
    """(folder, participant, session, labelled minutes) for every labelled recording."""
    out = []
    for folder in sorted(glob.glob(os.path.join(SESSIONS, "LIVE_*"))):
        lab = os.path.join(folder, "labelled.csv")
        runners = glob.glob(os.path.join(folder, "*_runner.csv"))
        if not runners:
            continue
        with open(runners[0], newline="", encoding="utf-8") as f:
            r = next(csv.DictReader(f), None)
        if not r:
            continue
        minutes = 0.0
        if os.path.exists(lab):
            with open(lab, newline="", encoding="utf-8") as f:
                head = f.readline().strip().split(",")
                k = head.index("label")
                minutes = sum(1 for line in f if len(c := line.rstrip("\n").split(",")) > k and c[k] != "none") / 50 / 60
        out.append((folder, r["participant"], r["session"], minutes))
    return out


def cmd_list(_):
    rows = recordings()
    people = {}
    for folder, p, s, m in rows:
        people.setdefault(p, []).append(f"{s} {os.path.basename(folder)} ({m:.1f} min)")
    for p, recs in people.items():
        print(f"{p:12s} " + "; ".join(recs))
    print(f"\n{len(people)} participants, {len(rows)} labelled recordings")


def rewrite_participant(path, old, new):
    """Replace the participant field (not other text) in a CSV that has a participant column."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or "participant" not in rows[0]:
        return False
    k = rows[0].index("participant")
    changed = False
    for r in rows[1:]:
        if len(r) > k and r[k] == old:
            r[k] = new
            changed = True
    if changed:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\n").writerows(rows)
    return changed


def cmd_pseudonymise(a):
    hits = [r for r in recordings() if r[1] == a.name]
    if not hits:
        sys.exit(f"No recording has participant '{a.name}'.")
    for folder, *_ in hits:
        for path in glob.glob(os.path.join(folder, "*.csv")) + glob.glob(os.path.join(folder, "*.csv.part*")):
            rewrite_participant(path, a.name, a.code)
            base = os.path.basename(path)
            if f"_{a.name}_runner" in base:
                os.replace(path, os.path.join(folder, base.replace(f"_{a.name}_runner", f"_{a.code}_runner")))
        print(f"{os.path.basename(folder)}: {a.name} -> {a.code}")
    os.makedirs(PRIVATE, exist_ok=True)
    new = not os.path.exists(ROSTER)
    with open(ROSTER, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["code", "name", "contact", "recording", "consent_time"])
        for folder, *_ in hits:
            w.writerow([a.code, a.name, "", os.path.basename(folder), "recorded before the study protocol"])
    print(f"Name kept only in {os.path.relpath(ROSTER, HERE)} (git-ignored). Note: git history still holds the "
          "old files; rewriting history is a team decision.")


def cmd_withdraw(a):
    hits = [r for r in recordings() if r[1] == a.code]
    if not hits:
        sys.exit(f"No recording has participant code '{a.code}'.")
    dest = os.path.join(SESSIONS, "_withdrawn")
    for folder, _, s, m in hits:
        if a.delete:
            shutil.rmtree(folder)
            print(f"deleted {os.path.basename(folder)} ({s}, {m:.1f} min)")
        else:
            os.makedirs(dest, exist_ok=True)
            shutil.move(folder, os.path.join(dest, os.path.basename(folder)))
            print(f"moved {os.path.basename(folder)} to sessions/_withdrawn/ ({s}, {m:.1f} min)")
    if os.path.exists(ROSTER):
        with open(ROSTER, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if rows and "withdrawn" not in rows[0]:
            rows[0].append("withdrawn")
            for r in rows[1:]:
                r.append("")
        k = rows[0].index("withdrawn")
        for r in rows[1:]:
            if r[0] == a.code:
                r[k] = time.strftime("%Y-%m-%d %H:%M") + (" deleted" if a.delete else " moved")
                if a.delete:                       # nothing left to find: drop what identifies them
                    r[1] = r[2] = ""
        with open(ROSTER, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\n").writerows(rows)
    print("Now retrain (Model/train.py) so the live model no longer contains this person, and commit the removal.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    p = sub.add_parser("pseudonymise"); p.add_argument("name"); p.add_argument("code"); p.set_defaults(fn=cmd_pseudonymise)
    w = sub.add_parser("withdraw"); w.add_argument("code"); w.add_argument("--delete", action="store_true")
    w.set_defaults(fn=cmd_withdraw)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
