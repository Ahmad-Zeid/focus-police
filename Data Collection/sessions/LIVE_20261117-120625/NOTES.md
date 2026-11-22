# LIVE_20261117-120625: S2 by P2, kept aside as TEST

Recorded 17 Nov 2026 with the guided runner as `S2` by participant P2 (Galaxy Watch SM-R870, left wrist,
right-handed, sitting). 11 blocks of 54 s: writing, drinking, hold_cup, phone_scroll x2, phone_call,
idle x2, typing, phone_text, hold_cup.

`role.txt` says `test`: `Model/train.py` keeps this session out of training and of every choice and scores
the final model on it once. P2 appears in no other labelled session, so this is the cross-person test the
course guide asks for ("use unseen people or conditions when claiming that the system generalizes").

No calibration step was recorded; `train.py` takes the calibration pose from the first 5 s of typing.
