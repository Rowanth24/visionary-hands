# Closed-vocabulary policy (medical core)

Defines **which words live in the closed-vocab word model** vs. which are handled by
**fingerspelling**. The system is "closed-vocab signs + fingerspelling fallback for
everything else" — this doc draws that line.

> **Status: candidate seed for advisor validation.** The *engineering rule* below is ours to
> own; the *final word list* is a **clinical + Deaf-advisor decision**, not an engineering one.
> Today the model ships a 5-word tech demo (bowling/computer/deaf/drink/yes); this replaces it
> with a medically-meaningful core to build G2/G3 against.

## What "closed-vocab" means

The word model classifies each sign into one of a fixed list (or abstains). It cannot output a
word outside the list. Anything off-list → the abstention layer fires → route to fingerspelling
(`should_fingerspell`). So the closed vocab only needs the **high-value recurring words**;
the long tail is spelled.

## Inclusion rule — a word earns a closed-vocab slot only if it scores high on all four

| Factor | Include when… |
|---|---|
| **Frequency** | it recurs in real triage/bedside exchange |
| **Stakes** | conveying it matters clinically (pain, allergy, breathe) |
| **Recognizability** | it's a reliably *distinct* sign a landmark model can classify |
| **Fingerspelling cost** | spelling it every time would be slow/error-prone |

**Excluded → fingerspelling** (fail a factor — usually frequency or proper-noun):
patient **names**, specific **drug names** (metoprolol, amoxicillin), rare diagnoses,
long/specialized terminology. Infinite + low-frequency → spell them.

### Recognizability caveat (lesson from the fingerspelling track)

Selection must consider **model confusability, not just clinical need.** The fingerspelling
work showed landmark models confuse signs that differ only by occluded fingers (n vs m). When
two *clinically distinct* words have near-identical landmark trajectories, including both
invites confident-wrong errors. Mitigations, in order: prefer a less-confusable synonym; if
both are essential, flag the pair for higher abstention; only then consider extra signal. A
word that the model can't reliably separate from another in the set is a *liability*, not a
feature.

## Sizing & phased growth (tied to the gates)

- **Now (demo):** 5 words — placeholder, not medical.
- **G2 core:** lock a **~50-word** medically-meaningful core (the seed below), ≥50 clips/word,
  signer-disjoint. This is the real Phase-1 vocabulary.
- **Later:** expand to ~100–200 only as data + accuracy support it; every addition re-checked
  for confusability against the existing set.

Bigger is not better: each added word raises confusion risk and dilutes per-word data. Grow on
evidence (it clears the inclusion rule *and* doesn't degrade neighbors on the G1 benchmark).

## Ownership & validation

- **Final list:** validated with Deaf ASL experts + the clinical/community partner. ASL is the
  confirmed target (per locked assumptions); the same advisors confirm the *citation form* of
  each sign so train/serve match.
- **Known ceiling:** ASL grammar uses non-manual (facial) markers the landmark/handshape model
  doesn't capture — keep the vocab to words whose *manual* sign is sufficient for meaning.
- **Form-factor:** seated, face-to-face (front camera) — vocabulary should suit that exchange.

## Candidate seed vocabulary (~70 words — advisor to validate)

Organized by category. Borderline items (conditions/meds) are flagged `→FS?` as likely
better fingerspelled.

**Core responses & function (~18):** yes, no, maybe, more, stop, finished, again, wait,
help, please, thank-you, want, need, have, not, understand, dont-understand, repeat

**Pain & symptoms (~19):** pain, hurt, dizzy, nausea, vomit, fever, cold, hot, tired, weak,
sick, cough, breathe, cant-breathe, blood, swell, itch, numb, headache

**Body parts (~16):** head, eye, ear, nose, mouth, throat, neck, chest, heart, stomach,
back, arm, hand, leg, foot, skin

**Needs & people (~11):** water, food, bathroom, medicine, sleep, blanket, doctor, nurse,
family, phone, interpreter

**Time & quantity (~9):** now, today, yesterday, how-long, when, minute, hour, much, little

**Conditions / medical (flag `→FS?`):** allergy, pregnant, diabetes →FS?, blood-pressure →FS?,
shot/injection, surgery →FS?, x-ray →FS?  *(conditions & procedure names lean fingerspelling)*

## How it plugs in

- The validated list becomes the word model's `labels.json`; the model trains on signer-disjoint
  multi-source clips for exactly these words (G2).
- Off-list input → abstention → `should_fingerspell` → fingerspelling for names/meds/OOV.
- The benchmark (`benchmark_g1.py`) scores per-word; confusable pairs surface in the confusion
  matrix and feed back into vocab/abstention decisions.
