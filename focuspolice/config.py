"""Project-wide constants. Everything that encodes a proposal decision lives here."""

# Signal processing
FS = 50.0          # Hz, all recordings are resampled to this rate
WINDOW_S = 5.0     # classifier window length
HOP_S = 2.5        # hop between windows for training and offline evaluation
LIVE_HOP_S = 1.0   # how often the live system makes a decision
MIN_PURITY = 0.8   # a window is used for training only if this share of it has one label

# Fine labels are what gets recorded (see docs/PROTOCOL.md).
# Each maps to the activity class the classifier predicts.
FINE_TO_CLASS = {
    "writing": "writing",
    "typing": "typing",
    "calculator": "calculator",
    "drinking": "drinking",
    "hold_cup": "drinking",      # look-alike: cup in hand, not a phone
    "hold_bottle": "drinking",   # look-alike: bottle in hand, not a phone
    "phone_typing": "phone",
    "phone_scrolling": "phone",
    "phone_calling": "phone",
    "phone_table": "phone",      # phone flat on the desk, used with one finger
    "idle": "idle",
}

CLASSES = ["writing", "typing", "calculator", "drinking", "phone", "idle"]

# Focus state per class. The proposal: writing/typing are work, phone/idle are
# drifting, and "drinking and calculating are part of the work and must not
# trigger a nudge". Drinking is neutral: it neither raises nor drains the counter.
FOCUS = {
    "writing": "on",
    "typing": "on",
    "calculator": "on",
    "drinking": "neutral",
    "phone": "off",
    "idle": "off",
}

# Nudge rule from the proposal: vibration after two minutes out of focus.
NUDGE_AFTER_S = 120.0
DRAIN_RATE = 1.0        # seconds of counter drained per second of on-task time
SMOOTH_WINDOWS = 3      # majority vote over this many consecutive decisions
NUDGE_MATCH_TOL_S = 60.0  # a nudge counts as correct if within this of a reference nudge


def focus_of(cls: str) -> str:
    """Focus state for a class name; unknown or unlabeled counts as neutral."""
    return FOCUS.get(cls, "neutral")
