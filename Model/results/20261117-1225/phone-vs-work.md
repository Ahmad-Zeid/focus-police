# Phone versus work: development comparison

Each fold trained on one session and scored on the other. S1 (P1, Galaxy Watch SM-R920) and S2 (P2, Galaxy Watch SM-R870) both used the left wrist. The S1 notes say the participant wrote with the watch hand. Read the aggregate scores as development results; there is no independent TEST session yet.

Compared 210 setups.
Best for individual activities: k-nearest neighbours, 4 s, acc+gyro, time
  activity macro-F1 0.443; exact activity accuracy 49.7%; phone vs work accuracy 84.3%
  S1 held out: caught 62.5% of phone windows; flagged 12.0% of work windows
  S2 held out: caught 97.0% of phone windows; flagged 14.9% of work windows
  balanced phone/work accuracy: 83.2%
Best for phone vs work: k-nearest neighbours, 4 s, acc, time
  activity macro-F1 0.401; exact activity accuracy 46.9%; phone vs work accuracy 85.8%
  S1 held out: caught 52.3% of phone windows; flagged 7.8% of work windows
  S2 held out: caught 96.0% of phone windows; flagged 8.6% of work windows
  balanced phone/work accuracy: 83.0%

Movement threshold baseline (2 s RMS acceleration): 45.8% accuracy, 54.5% balanced accuracy.

The baseline uses RMS user acceleration over 2-second windows. In each fold, the threshold and direction were picked by balanced accuracy on the training session only. Balanced accuracy averages phone recall and work recall.
