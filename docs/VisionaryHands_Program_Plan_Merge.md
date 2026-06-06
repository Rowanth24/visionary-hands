# Visionary Hands — Unified Program Plan (Merge of the Two Repos)
**ASL → Text, toward on-device (Jetson Nano) deployment**
Date: 2026-06-04 · Companion to `VisionaryHands_Analysis_and_NextSteps_2026-06-01.md`

> Planning document. No code/repo changes performed. All accuracy figures below are read directly from saved benchmark/metrics JSON in the two repos and verified.

---

## Addendum A — Locked decisions & council/pathfinder findings (2026-06-04)

**Product scope (LOCKED):**
- **One deployment context: medical / healthcare ASL communication** — chosen for impact. Closed-vocabulary, offline, two-way.
- **Primary users: ASL signers in Suriname (South America).** ⚠️ Verify the actual sign language/variant used there at G2 — do **not** assume it equals US ASL; validate vocabulary with local users/advisors before training.
- **Positioning: AUGMENTATION, not replacement.** Per Cameo Hunsaker's Interpreter Shortage Study (Appendix B), the shortage is mainly a pay/credentialing/policy problem — so the tool is a gap-filler for when a qualified interpreter is unavailable, never a substitute. This is the ethical stance *and* the differentiator from OmniBridge (explicitly not competing).
- **Advisors / Deaf feedback loop:** Cameo Hunsaker (AZ Commission for the Deaf & HoH) + community partner Chad Price.

**New gate — G0.5 Deaf-user feedback (added, early):** before scaling vocabulary or modeling, put the current closed-vocab demo in front of a Deaf ASL user/advisor; capture (a) is it useful, (b) what medical vocabulary matters, (c) what feels wrong/reductive. Re-run lightly at each milestone. *(Was: validation deferred to Fall.)*

**Abstention = shipping precondition (ELEVATED from a Phase-4 "win"):** the live path MUST say "not confident — please repeat / fall back to fingerspelling" below a calibrated threshold. A confidently-wrong output in a medical exchange is worse than none.

**Fingerspelling-FIRST (promoted to primary near-term track):** data approach in Appendix C. Especially apt for medical (names, meds, terminology are routinely fingerspelled) and is the OOV fallback for the closed vocab.

**Pathfinder findings folded in:**
- **G-A — model lineage:** the oracle is `InceptionI3d` (`Vanilla…/code/pytorch_i3d.py:152`) **only**. The EPICS Kaggle `i3d_r50` line is **RETIRED entirely** (different architecture, ~0.16, weak in team testing) — not oracle, not teacher.
- **G-B — first merge experiment:** before any "landmark vs vanilla" verdict, run the **EPICS dual-hand landmark transformer** (`landmark_baseline/model.py:9`, 58%/5-cls) through the **Vanilla benchmark harness** head-to-head vs `InceptionI3d`. The prior "landmark loses (0.45)" used the Vanilla repo's *weaker* encoder (`landmark_utils.py:102` `HolisticSequenceEncoder`), not this one — verdict invalid until redone.
- **G-C — preprocessing standard:** one module for train+serve = EPICS `landmark_baseline/preprocess.py` (dual-hand, handedness-canonical). Retire/adapt `landmark_utils.normalize_landmark_points` to avoid train/serve skew.
- **G-D — reuse, don't rebuild:** multi-source data + adaptation partly exists — `prepare_cross_source_finetune_data.py` (`build_wlasl_index`, `build_aslense_index`, `download_video`) + `train_vanilla_cross_source.py` (`anchor_regularization`, EWC-style). Build G2/G4 on these.
- **Integration seam:** add a `LandmarkBackend(BaseASLBackend)` to `asl_backends.py` so the existing harness judges both models with near-zero new eval code.

**Pooling is a hypothesis (council):** before harmonizing 5 datasets, prove the smallest version at G2 — train WLASL+MS-ASL, test held-out ASL Citizen, per-source.

**Objective hierarchy (accuracy-first — UPDATED 2026-06-05):** the **north star is model accuracy / real-world generalization.** Hardware (**Jetson Orin Nano**) is a **secondary, tunable constraint used lightly — NOT a gate.** We tune the hardware/runtime (TensorRT, quantization, offload) to support the most accurate model, within reason. This **supersedes the latency-gated framing** in §4 fork #1 and Phase 3/G3: measure latency to *plan hardware tuning*, not to *decide the model*. If `InceptionI3d` (or any heavier model) is materially more accurate than the landmark model, it remains a valid deployment choice and we adapt the hardware to it.

**Advisor input & locked assumptions (UPDATED 2026-06-05)** — folds in the Cameo Hunsaker + EPICS–ACDHH advisor meetings (transcripts archived in `~/Downloads/`):

*Findings folded in:*
- **Compete with VRI, not with human interpreters.** Video Remote Interpreting dominates hospital/legal settings and fails constantly (wifi/freezing; deaf patients wait days–weeks; sometimes never get an in-person interpreter — often illegally). The validated wedge = an **offline / no-wifi medical stopgap that beats failing VRI** when no in-person interpreter is available. Cameo strongly endorsed offline operation.
- **Form-factor constraint:** a front-worn camera only supports **seated, face-to-face** exchange; medical staff move around (behind the patient, at a computer). Design for this or state it as a known limitation.
- **Linguistic depth:** ASL has 5 parameters (handshape, location, movement, palm orientation, **non-manual/facial markers**); landmark/handshape models miss non-manual grammar — a known accuracy ceiling. Fingerspelling-first sidesteps part of it.
- **Fingerspelling was the prior glove's proven core** (per-letter accuracy + spelling a name) — reinforces the fingerspelling-first decision.

*Locked assumptions (proceed this semester without further user studies — team decision 2026-06-05):*
- **ASL is the confirmed target language** for the Suriname community partner and users. (Advisor noted Suriname has local sign languages — SSL and the Kajana variant — and that ASL is often only an international lingua franca; this risk is **acknowledged and set aside by decision**: the partner communicates in ASL and is building for ASL awareness.)
- **The engineering plan is treated as advisor-approved** — assume thumbs-up from Cameo Hunsaker (ACDHH), Chad Price (community partner), and the Deaf experts / ASL linguist.
- **No additional semester of user-needs interviews** — proceed on advisor input already gathered.

*Gate adjustment:* **G0.5 is downgraded** from "embed Deaf experts + user-research gate" to **"advisor approval assumed; apply already-gathered advisor input; lightweight check-ins only."** No longer a blocking gate.

---

## 0. How the two repos relate (the question answered)

| | `EPICS_Project` | `Vanilla Model ASL/Sign-Language-Recognition` |
|---|---|---|
| Origin | **100% yours** — your Kaggle scripts + their outputs | **Cloned upstream** WLASL I3D starter kit |
| Upstream (not yours) | — | `pytorch_i3d.py`, `start_kit/`, `Computer_Vision_Project.pdf`, vanilla I3D checkpoints |
| **Your** contribution in it | the whole repo | the `benchmark/` harness + testing + result JSONs |
| Core asset | **Edge-deployable landmark pipeline** (MediaPipe → compact transformer → ONNX/TorchScript), data tooling, Kaggle training | **Rigorous evaluation harness** (fixed multi-source manifests, per-source breakdown, shortlist eval, cross-source adaptation), strong in-distribution vanilla I3D |
| Best measured result | 58.3% test, 5 classes (landmark, dual-hand) | 0.95 in-distribution / 0.45 open-set multisource (vanilla I3D) |
| Deployable to Jetson? | **Yes** (light, ONNX) | **No** (heavy I3D) — accuracy oracle only |

**They are two halves of one system.** One has the *judge* (benchmark) and the *accuracy oracle* (vanilla I3D). The other has the *deployable model* (landmark) and the *data pipeline*. Neither alone is the product.

---

## 1. The "best of both worlds" thesis

Combine the four assets into **one repo, two model tracks, one shared judge, one data pipeline**:

```
                    ┌─────────────────────────────────────────┐
                    │   UNIFIED DATA PIPELINE (multi-source)    │
                    │  WLASL + aslense + MS-ASL + ASL Citizen   │
                    │  + your phone clips → harmonized vocab     │
                    │  emits: (a) video clips  (b) landmark arrays│
                    └───────────────┬───────────────┬───────────┘
                                    │               │
                  ┌─────────────────▼──┐        ┌───▼──────────────────┐
                  │ ORACLE / TEACHER    │        │ EDGE / STUDENT       │
                  │ Vanilla I3D         │        │ Landmark transformer │
                  │ (accuracy ceiling,  │        │ (ONNX/TensorRT,      │
                  │  not deployed)      │──distill──▶ Jetson, ≤1.5s)    │
                  └─────────────────┬──┘        └───┬──────────────────┘
                                    │               │
                    ┌───────────────▼───────────────▼───────────┐
                    │   ONE BENCHMARK HARNESS (the judge)         │
                    │  per-source top-1/top-5, open-set +         │
                    │  shortlist mode, confidence/abstention,     │
                    │  Jetson latency                             │
                    └───────────────┬────────────────────────────┘
                                    │
                    ┌───────────────▼────────────────────────────┐
                    │  DEPLOYMENT                                  │
                    │  closed-vocab mode (shortlist) → demo        │
                    │  + temporal voting + TTS (2-way)            │
                    └─────────────────────────────────────────────┘
```

Role assignment, justified by the evidence:
- **Vanilla I3D = oracle + teacher, not the deployed model.** It's strong in-distribution (0.85–0.95) but heavy and overfit to WLASL's domain (aslense = 0.10). Use it as the accuracy ceiling and, later, as a **distillation teacher** for the landmark model.
- **Landmark model = the deployable edge model.** Appearance-invariant (key to phone/cross-source generalization) and the only thing that fits a Jetson Nano.
- **Benchmark harness = the single judge** both models report to. Promote it to first-class; make both models pluggable backends.
- **Shortlisting = a deployment *mode*** (closed-vocab), not a generalization fix.
- **Distillation = the bridge** that tries to give the edge model the oracle's accuracy.

---

## 2. Verified scoreboard (current state, the honest baseline)

Same vanilla checkpoint, different test sets (n=20 each — small, see gate G1):

| Test set | Vanilla top-1 | Per-source |
|---|---|---|
| `ollama_vs_best` (in-distribution WLASL) | **0.95** (19/20) | — |
| `wlasl20_v2` (in-distribution) | 0.85 | — |
| `vanilla_only` | 0.70 | — |
| `multisource20` **open-set** | **0.45** (9/20) | WLASL **0.80** / aslense **0.10** |

Levers tested:

| Lever | Effect | Keep? |
|---|---|---|
| Candidate-label **shortlist** | 0.45→**0.75** (multisource), 0.85→**1.0** (wlasl20) | ✅ deployment mode |
| **Ollama hybrid** | 0.45→0.45, 0.85→0.85 (net zero) | ❌ kill or re-spec |
| MediaPipe normalization | 0.45→0.45 | ❌ no effect |
| Landmark-only (this repo) | 0.45 vs vanilla 0.75 closed-set | ⚠️ undertrained here |
| Cross-source fine-tune | aslense val 0.125→**0.333** | ⚠️ only thing lifting weak source |

**Takeaway:** in-distribution ≈ 0.85–0.95, real cross-source ≈ 0.45, worst source ≈ 0.10. The program's north-star metric is **per-source open-set top-1 (lift the worst source)** + **closed-vocab top-1** + **Jetson latency** — not the bare 0.95.

---

## 3. The plan (phased, evidence-gated)

Each phase has an **exit gate** — a measurable criterion. Do not advance until the gate is met. This prevents the architecture-churn that stalled both repos.

### Phase 0 — Unify the workspace (1–2 days)
- New repo `visionary-hands/` that is cleanly **yours**:
  - `pipeline/` ← landmark extraction + dataset prep (from `EPICS_Project`)
  - `models/landmark/` ← compact transformer + export/bench (from `EPICS_Project`)
  - `eval/` ← the benchmark harness + manifests (your `benchmark/` work from the clone)
  - `oracle/` ← thin wrapper that loads the vanilla I3D **as an external dependency** (vendor or submodule the upstream WLASL starter; **keep its LICENSE + attribution**; do NOT absorb `Computer_Vision_Project.pdf` or other authors' artifacts as your own)
  - `deploy/` ← Jetson runtime, TTS
- `git init` + `.gitignore` (`.venv*`, `*.pt`, `*.pth`, `*.onnx` optional, large `data/`, `*.pyc`, `.DS_Store`). Reclaim ~7 GB from `EPICS_Project` duplicates.
- One `requirements`/env story: the repo already needs **Python 3.11** (torch/cv2); keep the split venvs note (MediaPipe vs ONNX protobuf conflict).
- **Gate G0:** both a landmark model and the vanilla I3D run from one repo, one command each.

### Phase 1 — Unify evaluation (the judge) (2–4 days)
- Promote `eval/` to the single harness. Extend the existing backend abstraction (`asl_backends.py` already has `--backend vanilla|ollama|...`) with a **`landmark`** backend pointing at the ONNX bundle, so both models score on identical manifests.
- Standardize every run to report: **per-source** top-1/top-5, open-set **and** shortlist, mean-confidence, and an **abstention rate** at a confidence threshold.
- **Grow the benchmark** (fixes n=20): ≥10–20 samples/label, ≥3–4 sources, **including a small set of your own phone clips**. (This is the only manual recording worth doing — for *evaluation*, not training.)
- **Gate G1:** a benchmark with ≥150–200 clips, ≥3 sources incl. phone, with per-source numbers and confidence intervals reported. No model decisions on n=20 again.

### Phase 2 — Unify data (the fuel) (1–2 weeks, the real lever)
- One multi-source pipeline that, from a single manifest, emits **both** video clips (for I3D) and **landmark arrays** (for the landmark model). Merge the `EPICS_Project` extraction scripts with the clone's `start_kit` data tooling.
- **Pool sources beyond WLASL**: aslense (already in your benchmark), **MS-ASL**, **ASL Citizen** (webcam/crowd-sourced — closest to phone domain), and segmented How2Sign if needed. Do **gloss/label harmonization** to a shared vocabulary (the unglamorous step that decides whether pooling helps).
- **Lock the demo vocabulary** here (~10–20 everyday conversational words; alphabet handled later as a separate fingerspelling mode + OOV fallback).
- **Gate G2:** a harmonized multi-source training set for the locked vocab with **≥50 clips/word** and a documented per-source/per-signer split that is **signer-disjoint** (kills the leakage that inflates in-distribution numbers).

### Phase 3 — Establish baselines on the unified benchmark (3–5 days)
- Run **both** models through G1's benchmark: per-source, open-set + shortlist.
- **Benchmark Jetson Orin Nano latency** for both (capture → MediaPipe → infer). Target: **Jetson Orin Nano** (Ampere, ~40 TOPS, 8 GB) — strong enough that a **TensorRT FP16/INT8 vanilla I3D may actually fit the ≤1.5 s budget**, so this is a real measurement, not a foregone result. For the landmark model the bottleneck is MediaPipe extraction, not the tiny transformer.
- Produce the **decision matrix**: accuracy (per source) × latency, per model.
- **Gate G3:** a 2-axis scoreboard exists. The deployed-model question (landmark vs distilled-vanilla vs vanilla) is now answered by data, not opinion.

### Phase 4 — Close the gaps (evidence-gated experiments) (2–4 weeks)
Run only experiments with a falsifiable hypothesis measured on G1's harness:
- **(a) Cross-source adaptation** of the deployable model on the pooled data — target: lift the worst source (aslense) materially above 0.33. *(Evidence says this is the lever that moves the weak source.)*
- **(b) Distillation** vanilla I3D → landmark — hypothesis: transfer oracle features to the edge model and beat the standalone landmark baseline. *(The "best of both worlds" experiment.)*
- **(c) Cheap inference-time wins (do first):** multi-clip test-time augmentation, **temporal voting** in the live loop, **confidence calibration + abstention** ("unsure — please repeat," critical for an assistive tool), and shortlisting in closed-vocab mode.
- **(d) Ollama hybrid:** give it one falsifiable job — re-rank vanilla's top-k **only when confidence is low**, using context priors — and measure. If it doesn't beat vanilla+shortlist on G1, **delete it.**
- **Gate G4:** at least one method beats `vanilla + shortlist` on per-source open-set on the grown benchmark, with CIs that don't overlap.

### Phase 5 — Ship demo + edge deployment (ongoing → Fall)
- **Closed-vocab demo now**: vanilla/landmark + shortlist (0.75–1.0) is genuinely demo-ready — ship it explicitly as a known-vocabulary mode.
- **Edge deploy** the latency-passing model to Jetson Nano (ONNX/TensorRT), wire **TTS** for the 2-way loop (and speech-to-text for the return path).
- **Honest metrics card**: per-source open-set + closed-vocab numbers; retire the bare "95%" from any external claim.
- **Validation**: schedule Deaf-community / ASL-user testing (the design doc's stated gap) for Fall.

---

## 4. Decision forks (yours to make; data resolves most)

1. **Deployed model** — landmark vs distilled-vanilla vs vanilla. *Resolved by Gate G3.* **Genuinely open** on Orin Nano: the board may run a TensorRT vanilla I3D in budget, so the choice is a real accuracy-vs-latency-vs-power tradeoff. Landmark still wins on latency, power, and cross-source generalization; vanilla wins on raw accuracy. Distillation only if a deployable landmark model has a large accuracy gap to vanilla.
2. **Repo home** — ✅ **DECIDED: new directory** (`visionary-hands/`), keeping the upstream-clone IP cleanly separated from your own work.
3. **Edge target** — ✅ **DECIDED: Jetson Orin Nano** (Ampere ~40 TOPS, 8 GB).
4. **Distillation investment** — only if Gate G3 shows a real accuracy gap between oracle and a deployable edge model.
5. **Vocabulary breadth** — ✅ aligned: aim broad/everyday, but admit words to "shipping" as they clear your accuracy bar per-source; rest → fingerspelling fallback.
6. **LLM role** — narrowed to: low-confidence top-k re-ranking (must prove itself) and/or gloss→fluent-English post-processing (separate layer, Fall+). Not recognition.
7. **Phone eval set** — ✅ DECIDED: yes. You will manually record a small phone-camera eval set (a few clips/word) for *measurement only* (not training), **recorded on demand** — needed at Gate G1 (seed the benchmark) and confirmed at G3. Claude will flag exactly when.

---

## Decision log + resume pointer
All decisions captured above are final as of 2026-06-04. **Status: planning complete, ready to execute at Gate G0.**
To resume after a compaction/new session: read this file + `VisionaryHands_Analysis_and_NextSteps_2026-06-01.md`. Start at **Phase 0 (G0)** — create the new `visionary-hands/` repo, git init + .gitignore, bring in the landmark pipeline (from `EPICS_Project`, yours) and the benchmark harness (from `Vanilla Model ASL/Sign-Language-Recognition/benchmark`, yours), and wire the upstream WLASL I3D as an external dependency. Target edge = Jetson Orin Nano. Nothing in this plan depends on un-saved conversation state.

---

## 5. Non-negotiables / hygiene
- **Per-source metrics always.** Pooled numbers hide the domain gap (0.45 = 0.80 WLASL / 0.10 aslense).
- **Signer-disjoint splits.** Random instance splits manufacture fake high accuracy (the likely cause of the 0.95).
- **Preserve checkpoints + manifests** (already Codex's discipline — keep it).
- **Respect the cloned repo's license**; carry forward only your additions; attribute upstream.
- **Don't retrain without a hypothesis** that can beat `vanilla + shortlist` on the grown benchmark.
- **Secrets**: Kaggle key lives in `~/.zshrc` plaintext — never commit it; rotate if shared.

---

## 6. The one-screen summary
1. Make one repo (yours): landmark pipeline + benchmark harness + vanilla-as-dependency. (G0)
2. Make the benchmark the single judge for both models; grow it past n=20 with phone clips. (G1)
3. Pool multi-source data (WLASL+aslense+MS-ASL+ASL Citizen+phone), harmonize, signer-disjoint splits, ≥50 clips/word. (G2)
4. Baseline both models on accuracy × Jetson latency → pick the deployed model by data. (G3)
5. Lift the worst source via adaptation + distillation + inference-time tricks; kill/re-spec the LLM hybrid. (G4)
6. Ship the closed-vocab demo + Jetson + TTS; plan Deaf-community validation.

---

## Appendix B — Interpreter Shortage Study (advisor context)
Source: *Parking Lot Debrief* podcast, Parts 1 & 2, with Cameo Hunsaker (Interpreter Professional Development Specialist & staff interpreter, Arizona Commission for the Deaf & Hard of Hearing). Transcribed via yt-dlp; key points:
- The ASL interpreter shortage is decades old but felt acutely now — "access is popular," so interpreters are placed in more settings while supply is flat/shrinking.
- **Root causes are structural, not technological:** (1) **pay** — interpreters' #1 cited cause; rates stuck ~$25–55/hr (most $40–50) from 2012 to present; (2) **credentialing gaps** — ADA requires only a "qualified" interpreter without defining a credential; left to states; unqualified people fill roles with no accountability (CPC binds only the certified).
- VRI/VRS made interpreting nationally connected in ways the "qualified" definition never anticipated.
- **Implication for this project:** the shortage is a pay/policy problem → the tool must be **augmentation, not replacement** (gap-filler when no qualified interpreter is available). This is both the ethics and the OmniBridge differentiator.
- Two related Google Drive files the user referenced could not be retrieved (Drive sign-in interstitial); revisit if shared as "anyone with link" or downloaded locally.

## Appendix C — Fingerspelling-first: the data recipe
**Reframe:** static fingerspelling needs *frames*, not *clips* → data is abundant and self-generable, dissolving the "no letter-clip datasets" concern.
- **Public image datasets:** ASL Alphabet (Kaggle, ~87k imgs, A–Z+space/del/nothing), Sign Language MNIST (~35k, 24 letters), Massey ASL.
- **Landmark conversion:** run MediaPipe Hands over the images once → 21-pt landmarks → train a small classifier (appearance-invariant → transfers to phone/Jetson far better than pixels).
- **Self-generation:** each sample is one frame, so a ~10 s hold at 30 fps ≈ 300 samples/letter; team + Deaf signers in Suriname can capture all 24 static letters across people/lighting/angles in <1 hr.
- **J & Z** are the only motion letters → exclude initially (24-class) or handle later as 2–3 frame mini-gestures.
- **Architecture:** shares the front-end with the word model — camera → MediaPipe Hands → standardized normalization (G-C) → MLP → 24 letters + "transition/none" + confidence threshold. One front-end, two heads (letter/word).
- **Letters → spelled words:** detect stable-hold frames (low landmark velocity) → emit letter; ignore transitions → assemble string → dictionary/edit-distance or small-LLM autocorrect. Gives OOV coverage for medical terms/names/meds.
- **Continuous (fast) fingerspelling:** harder; one real dataset = ChicagoFSWild/+. Defer.
- **Medical fit:** fingerspelling carries names, medications, and terminology that have no sign — so fingerspelling-first is especially valuable for the medical scope.
