# Participant study: protocol

Focus Police is evaluated as a small study, using the experiment-design principles from the course (week 7: protocols, labelling, ground truth, validation, reproducibility; the walking-speed example in class). The goal is to measure how well the model works on **a person it has never seen**, which is the weakness the instructor asked us to address.

## Research question

Can a wrist-worn motion sensor tell phone use from desk work reliably on a new person, and does calibrating to each person's resting pose help?

## Variables

| Kind | Here |
|---|---|
| Independent (manipulated) | The activity the participant is asked to do: 5 work activities (typing, writing, reading/idle, drinking, holding a cup) and 3 phone activities (scrolling, texting, holding to the ear). Analysed offline: calibration on/off. |
| Dependent (measured) | Phone caught (share of phone time flagged), false alarms (share of work time flagged), false nudges, time from picking up the phone to the nudge. Activity accuracy is secondary. |
| Controlled (kept constant) | Same watch (Galaxy Watch5 Pro) and app, left wrist, strap snug, 50 Hz, sitting at the same desk and chair, the same 8 activities for 60 s each, the same instructions read from the screen, the same 5 s calibration pose (hands resting on the keyboard). |
| Additional measures | Writing hand, strap fit, posture, a free-text note; the calibration pose angle (logged automatically). |
| Counterbalancing | The activity order is shuffled per participant (seeded by their code), so practice or tiredness doesn't always fall on the same activity. |

## Ground truth and labels

The runner records the start and end of every activity on the watch's own clock, so the labels come from the protocol, not from guessing afterwards. The first 3 s after each start and the last 3 s before each end are trimmed (`segment.py`). `check.png` in each recording shows the labelled ranges on the signal for a visual check.

## Setup (once per laptop)

Python 3.10 or newer, from the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

The watch app: install it once with `FocusWatch/WATCH-SETUP.md`. Then start the receiver and leave it running:

```bash
cd WristStream && python3 receiver.py
```

It prints the laptop's address; type it into the watch app's server field (the laptop and watch must be on the same Wi-Fi). The runner is at `http://localhost:8000/runner`.

## Reproduce the results without a watch

Every recording used is in `sessions/`, with its labels. From the repository root:

```bash
cd Model && python3 train.py --test-participant P-XXXX --no-deploy
```

This rebuilds the comparison and the report (`Model/REPORT.md`) from the raw recordings; `--no-deploy` leaves the live model untouched. To watch the live page without a watch, replay a recording (README, "replay").

## Procedure (about 12 minutes per person)

1. Watch on the left wrist, strap snug. Tap START on the watch. Open `http://localhost:8000/runner`, session **STUDY**. The page gives a fresh random code (P-XXXX).
2. Optionally type the person's name and contact in the private roster box (see below).
3. **Consent screen:** read it aloud. Tick 18+ and agreed only if both are true. "They declined" discards everything and generates a new code.
4. **Calibration:** 5 s, hands resting on the keyboard, still.
5. **8 activities × 60 s**, in this person's order. "Redo activity" if something went wrong (phone rang, they misunderstood).
6. **Debrief:** thank them, write their code on a slip for them, tap STOP on the watch.

## Ethics and privacy

Small scale, but handled as a real study would be:

- **Informed, voluntary consent**, read aloud before anything is recorded: purpose, what is recorded, how long, who sees it, the right to stop at any time and to withdraw later. 18+ only. Consent is logged in the recording as an event, without a name.
- **Data minimisation:** only the watch's motion sensors. No audio, video, location, health data or phone content. Additional measures are limited to what the analysis uses.
- **Pseudonymisation:** recordings carry only a random code (`participant` column, runner file name). Names and contacts, if collected at all, go to `Data Collection/study-private/roster.csv`, which stays on the experimenter's laptop and is git-ignored. The two files are never joined in the analysis.
- **Withdrawal:** with the person's code (or a roster lookup), `python study.py withdraw P-XXXX` removes their recordings from the study (`--delete` removes them for good and blanks their roster row); then retrain so the live model no longer contains them.
- Recordings made before this protocol (22 Nov) were pseudonymised with `study.py pseudonymise`. Git history still contains the earlier file names; rewriting history is a team decision.

## Analysis

`Model/train.py` treats STUDY recordings as training data. With 3 or more people it compares models by **leaving one participant out at a time**: every score is on a person the model never saw. For the final number, set people aside before anything is tuned:

```
python train.py --test-participant P-XXXX[,P-YYYY]
```

The report lists the score for each held-out person and the ablation (what calibration, wrist angle and context each add) on the test people.

## Tools

```
python study.py list                      # who took part (codes only), recordings, minutes
python study.py withdraw P-XXXX [--delete]
python study.py pseudonymise NAME P-XXXX  # for recordings made with a name as the ID
```

## Limitations to state in the report

A handful of participants, mostly students, one watch, one desk, scripted activities, one sitting each. Results show whether the approach transfers to new people; they don't establish population-level accuracy.
