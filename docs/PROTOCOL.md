# Recording protocol

Read this before recording anything. Data that does not follow it may not be usable,
and there is no time to record twice. Deadline: November 17, 23:55.

## Before the full recording day: 5-minute test

1. Record 5 minutes on the watch: 1 minute each of typing, writing, phone scrolling,
   idle, and sipping from a bottle.
2. Send the files to the team the same day.
3. We check: sample rate (need 25 Hz or more, 50 Hz preferred), accelerometer **and**
   gyroscope both present, gravity present in the acceleration, timestamps sane.

Do not start the full recording until this test has passed.

## Setup, the same for every session

- Watch on the **writing hand**, snug, same position each time. Note it if someone
  cannot do this.
- Accelerometer and gyroscope, 50 Hz if the app allows it.
- Webcam or phone camera filming the desk and both hands for the whole session.
- **Three hard taps of the watch hand on the desk** at the start and again at the end,
  in view of the camera. This is how the video is lined up with the signal.
- Take the watch off and put it back on between sessions.
- One folder per session. Never append to an earlier recording.

## What to record, per person

At least 3 people. At least 2 different desks across the data set.

### Two scripted sessions (about 22 minutes each, different desks if possible)

Two minutes of each block, in any order. A helper calls the changes.

| Label | What the person does |
|---|---|
| `writing` | Handwriting on paper |
| `typing` | Typing on a laptop keyboard |
| `calculator` | Using a handheld calculator on the desk |
| `drinking` | Sipping from a bottle or cup every 15 to 20 seconds, hand back on the desk between sips |
| `hold_cup` | Holding a cup in the watch hand while reading, no phone |
| `hold_bottle` | Holding a bottle in the watch hand while reading, no phone |
| `phone_typing` | Phone in hand, typing a message |
| `phone_scrolling` | Phone in hand, scrolling a feed |
| `phone_calling` | Phone held to the ear |
| `phone_table` | Phone flat on the desk, used with one finger |
| `idle` | Sitting still, hands resting, not working |

### One unscripted study session (20 to 30 minutes)

The person actually studies. They are told to take **at least two phone breaks longer
than two minutes**, a few short phone checks, and to drink when they want. Nobody calls
the changes. This is the only data where "false nudges per hour" means something.

## Labels

Either method is fine. Use the same label names as the table above.

**A. Streaming through the laptop (preferred if the watch can stream).**
Open `http://<laptop-ip>:8765/record` on a phone, fill in the session details, press
Start, and tap the label button each time the activity changes. The session folder is
written when you press Stop.

**B. Recording on the watch, labelling from video afterwards.**
Write a `labels.csv` per session with columns `start,end,label`, in seconds from the
start of the recording, then run `python -m focuspolice.importer` (see the README).
Use the desk taps to find the offset between video and signal.

## Session folder

```
data/raw/<person>_<session>/
    imu.csv      t,ax,ay,az,gx,gy,gz
    labels.csv   start,end,label
    meta.json    person, session, kind (scripted | unscripted), wrist, desk, device
```

Keep the original exports and the videos too. The course asks for raw signals,
timestamps and reference labels to be kept.
