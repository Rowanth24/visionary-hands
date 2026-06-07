"""The 24 static ASL fingerspelling letters (J and Z are dynamic — excluded in Phase 1)."""

LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m",
           "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y"]

assert len(LETTERS) == 24 and "j" not in LETTERS and "z" not in LETTERS
LETTER_TO_INDEX = {letter: i for i, letter in enumerate(LETTERS)}
