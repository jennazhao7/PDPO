# Stage 2 — Randomized-Response Arms: PRE-REGISTRATION

Recorded **2026-08-19, before any arm is launched.** No result exists yet. Nothing below may be
tuned after seeing data.

## Prediction (pre-registered)

1. `signed_diff` member/non-member separation scales as **(1 − 2γ) = tanh(ε/2)**, γ = 1/(1+e^ε).
2. `magnitude` separation is approximately **flat in ε**.
3. At **ε = 0**: `signed_diff` → chance; `magnitude` → unchanged from the clean control.

Attenuation factors implied, computed and verified to 1e-12:

| ε | γ = 1/(1+e^ε) | tanh(ε/2) = 1−2γ |
|---:|---:|---:|
| 0.0 | 0.500000 | **0.000000** |
| 0.1 | 0.475021 | 0.049958 |
| 0.5 | 0.377541 | 0.244919 |
| 1.0 | 0.268941 | 0.462117 |
| 2.0 | 0.119203 | 0.761594 |
| ∞ (clean) | 0.0 | 1.000000 |

## Arms — 6 (5 RR + 1 reused control), frozen

| # | Arm | ε | γ | Source of ε |
|---|---|---:|---:|---|
| 1 | clean control | ∞ | 0 | Stage 1, reused (protocol-identical) |
| 2 | RR pure coin flip | **0.0** | 0.500000 | pre-registered anchor; tanh(0)=0 |
| 3 | RR | **0.1** | 0.475021 | PROPS, same tables |
| 4 | RR | **0.5** | 0.377541 | PROPS (arXiv 2508.06783v2), Tables 1/2/3/7/8/9, Fig 4 |
| 5 | RR | **1.0** | 0.268941 | PROPS, same tables |
| 6 | RR | **2.0** | 0.119203 | PROPS, same tables |

**ε sourcing.** PROPS reports experiments at **ε ∈ {0.1, 0.5, 1.0, 2.0, ∞}** and defines
"the label ℓ_RR is flipped with probability γ_ϵ = 1/(1+e^ϵ)" — identical to our formula.

**PROPS cites no prior RR-based label-DP alignment work**, stating "to the best of our knowledge,
the concept of Label Differential Privacy … has not been explored in the context of alignment."
So PROPS is the *only* available source in this literature, and the three literature ε values are
drawn from its set.

**AMENDMENT 2026-08-19 — ε = 0.1 RESTORED. My exclusion was wrong.**

I originally excluded ε = 0.1 on the grounds that tanh(0.05) = 0.0500 sits within 0.05 of the
ε = 0 anchor and would duplicate a curve point. That reasoning is correct on the **attenuation
axis** and wrong on the **utility axis**, which is the one that matters here.

At ε = 0 the labels are pure coin flips: there is no utility to defend, so that arm cannot speak
to whether a *usable* model still leaks. At ε = 0.1 PROPS reports a working configuration. So
ε = 0.1 is the single cell where attenuation is near-total **and the source paper claims the
configuration works** — which makes it the headline cell of the comparison, not a redundant one.

Arms are therefore **ε ∈ {0, 0.1, 0.5, 1.0, 2.0}**, five RR arms plus the reused clean control.
Recorded as a reversal, with date, rather than swapped in silently.

## Mechanism — label flip only

- Per training pair, with probability γ, **swap** which response occupies `chosen` and which
  `rejected`. Prompt and both response texts unchanged; both still receive gradient.
- **No** examples dropped. **No** noise on logits, gradients, or weights. **No** DP-SGD.
- Flip drawn **once at data-prep**, fixed per example for the whole run.
- Per-example boolean `was_flipped` persisted in the manifest.

## Evaluation — scored against the TRUE, UNFLIPPED label

The membership statistic is computed with `chosen`/`rejected` in their **true PKU v3 order**,
before RR. The model is trained on flipped labels; the attack reads true labels.

Concretely: the MIA member set is the **unflipped** `data/pku_saferlhf_secure_v3/train_pref.jsonl`;
the flipped file is used **only** as trainer input. Scoring against the flipped label would
invalidate the experiment.

## Protocol — identical to Stage 1

PKU v3, 8000 train / 3000 test, `lora_r=16`, `beta=0.5`, `lr=2.5e-05`, seed 42, epochs {1,2,3},
checkpoints at 25/50/75/100%, FP32, same trainer, same eval code, same attack code.
**One seed.** A second seed only on request after the first read-out.

## Metrics

AUC and **TPR@1%FPR** with bootstrap CIs, for `signed_diff`
(`reference_calibrated_dpo_margin`) and `magnitude` (`abs_reference_calibrated_margin`).
The number of non-member examples defining each FPR threshold is reported beside every TPR.
Losses below 1e-3 in scientific notation.

**TPR@0.1%FPR is NOT reported.** See the blocking decision below.

## Fit-matching

RR arms are matched to the Stage 1 clean curve on **measured train/held-out gap**, never on epoch
fraction. HH-RLHF established that epoch fraction does not transfer across conditions
(HH loss 1.4514 → 0.0055 versus PKU e3 0.7684 → 0.0914 over the same fractions).

## Stop conditions

No widening of the sweep, no added ε values, no added seeds, no hyperparameter changes on my
initiative. If an arm fails or a result looks wrong: stop and report.


---

## AMENDMENT 2026-08-19 — TPR@0.1%FPR retired PROJECT-WIDE

Not merely uncited: **retired**. At n = 3000 non-members the 0.1% threshold is set by **3**
scores, giving a 95% CI on the realized FPR of [0.00000, 0.00213] — a 226% relative width whose
lower bound touches zero. Reaching the same 71% relative width that 1% has at n = 3000 would need
~30,000 non-members, i.e. ~20 hr of evaluation per checkpoint (~960 GPU-hr, ~$336 across this
stage) — over three times the whole Stage 2 cap.

**This retires the historical low-FPR figures too**, including the rDPO 0.05–0.23 versus
RE-DPO 0.55–0.65 comparison. Those were computed on **500**-non-member pools, where 0.1% FPR is
defined by **0.5 examples** — strictly worse than the case that prompted this decision. That
comparison must not appear in the paper in any form.

Tightest FPR reported anywhere: **1%**, always with the count of non-members defining the
threshold printed beside it.

## AMENDMENT 2026-08-19 — budget baseline is UNVERIFIED

Billed spend could not be queried: the Cloud Billing Budget API is not enabled on `jenna-pdpo`,
the account returns a permission error for this user, and no BigQuery billing export exists.
On explicit instruction, Stage 2 launches against the **ledger estimate** — approximately $53
spent, approximately $147 headroom against the $200 ceiling — and that figure is recorded here as
**an unverified assumption, not a measurement**. It excludes the token-concentration VM, which ran
outside the ledger from at least Aug 19 02:54 until deletion at Aug 19 ~17:10, across 2,549
supervisor relaunch attempts.

Revised projection with five RR arms: 5 × 38.05 = **190.25 GPU-hr ≈ $66.59** against the $100 cap.

---

## AMENDMENT 2026-08-21 (A0.1) — PRIMARY ANALYSIS AXIS IS REALIZED FLIP RATE, NOT NOMINAL ε

**Recorded 2026-08-21, before any arm has read out.** Four ε=0 checkpoints are committed; no
attenuation figure has been computed against any axis, and no arm-level result exists. This
amendment therefore precedes the first read-out, which is the condition that makes it a
pre-registration rather than a post-hoc choice.

**The fact that forces it.** `stable_uniform(seed, id)` does not take ε. All five arms threshold
the **same 8,000 uniforms**, so the realized flip rates are not five independent draws — they are
one draw read at five thresholds. Verified exactly: `count(u < γ_ε)` equals `rr_flip`'s committed
flip count at every ε. Seed 42 drew high (mean u = 0.5074, +2.3σ; KS p = 0.118, χ²(19) p = 0.065 —
neither rejects uniformity), which depresses **every** realized rate below nominal γ in a perfectly
correlated way. Across 21 seeds at ε=2.0 the deviation has mean z = +0.015, sd = 1.20, so the
generator is sound and this is a property of seed 42's sample, not of the mechanism.

**Consequence.** Fitting observed attenuation against **nominal** γ would charge the mechanism for
a sampling accident of the seed. Fitting against the **realized** rate is exact and makes the seed
draw irrelevant to the mechanism claim.

**Decision.** The attenuation curve is fitted against **realized flip rate p**, prediction
`1 − 2p`. Nominal ε and `tanh(ε/2)` remain in every table because ε is the quantity PROPS reports
and the axis on which this work is comparable to it. **Every results table carries BOTH columns.**

| ε | realized p | realized 1−2p | nominal γ | nominal tanh(ε/2) |
|---:|---:|---:|---:|---:|
| 0.0 | 0.48950 | **0.02100** | 0.50000 | 0.00000 |
| 0.1 | 0.46400 | **0.07200** | 0.47502 | 0.04996 |
| 0.5 | 0.37200 | **0.25600** | 0.37754 | 0.24492 |
| 1.0 | 0.26038 | **0.47925** | 0.26894 | 0.46212 |
| 2.0 | 0.10850 | **0.78300** | 0.11920 | 0.76159 |

Realized p is `flips / 8000` from `data/stage2_rr/rr_audit.json`, reproduced bit-exactly by a
fresh run of the production flip function (C1d). **ε = 2.0 is where the two axes diverge most**
(0.78300 vs 0.76159, 2.8% relative); it is the cell where the axis choice is load-bearing.

**Not changed by this amendment:** the arm list, ε values, seed, γ formula, protocol,
hyperparameters, or any statistic definition. This fixes which x-axis the curve is fitted on.

---

## AMENDMENT 2026-08-21 (A0.2) — FLIP ASSIGNMENTS ARE NESTED ACROSS ε, BY CONSTRUCTION

Because the same 8,000 uniforms are thresholded at every ε, and γ is monotonically decreasing in
ε, the flipped sets are **strictly nested**:

```
flipped(ε=2.0) ⊂ flipped(ε=1.0) ⊂ flipped(ε=0.5) ⊂ flipped(ε=0.1) ⊂ flipped(ε=0.0)
       868            2083             2976            3712            3916
```

This is a property of the design, not an artifact and not a defect.

**It yields a matched design.** Cross-arm comparisons are **paired at the example level**: the same
example carries a known flip status in every arm, so arm-to-arm differences are within-example
rather than between independent samples. That is *more* statistically powerful than independent
draws per arm, because example-level variance in prompt difficulty and response length is
differenced out. It is the same reason common random numbers are used deliberately in simulation.

**Stated here, and to be stated plainly in the paper's setup.** A reviewer who discovers nesting
unannounced will read it as a bug. Recording it in advance is the difference between a design
choice and an accident.

**Corollary already verified (C3c).** Nesting is also why a reference cache built for one arm
cannot silently serve another only *because* some row differs: arms share orientation on most
rows, and refusal requires at least one differing row. Arms differ on thousands of the 8,000, and
cache paths are per-arm, so this is unreachable in practice — but the guarantee is per-row, and
P1b now writes ε into the cache and asserts it on load so the guarantee is stated rather than
incidental.

---

## AMENDMENT 2026-08-23 (A3) — THE `loss` COLUMN IS ONE TRAINING EXAMPLE, AGAINST FLIPPED LABELS

Confirmed by code, not inferred. `train_matched_cached.py:573` computes
`loss = F.softplus(-effective_margin).mean()` where `margin = beta * ((chosen_lp - ref_c) -
(rejected_lp - ref_r))` is built from a batch drawn from `args.data`; line 586 assigns
`last_loss = float(loss.detach().cpu())`; line 419 writes that verbatim into
`checkpoint_state.json`. Two consequences follow directly:

1. **Flipped labels.** For RR arms `args.data` is the swapped file, so the recorded loss is the
   objective on the *flipped* labels. It is **not comparable to Stage 1's clean loss column**, and
   the Gate 1 loss band does not apply to RR arms.
2. **One example, not an average.** With `--bsz 1` the microbatch is a single row, so `.mean()`
   averages over one element. The column is a single-example loss at the checkpoint step.

This fully explains the observed non-monotonicity (`0.6965 -> 0.5761 -> 0.8185 -> 0.6369` for
eps=0 e1) and values above ln 2 = 0.693 mid-training: those are independent single-example draws,
not optimisation anomalies. **The column is near-uninformative as recorded and must not be used as
a fit measure.** Averaged loss, recomputed from persisted margins, is the correct quantity.

Also resolves the open item in `analysis_cpu/SUMMARY.md` ("e3_p075 loss recorded as exactly 0.0000
— check the raw value"): it was one example scoring ~0. That cell's averaged train loss is 0.0208.

## AMENDMENT 2026-08-23 (B2/C2) — GATE 1 RECOUNTED OVER AVERAGED LOSS: 2 → 4

Because the loss criterion was applied to single-example noise, Gate 1 was recomputed over mean
DPO loss on the full 2,000-row train subsample, from margins already persisted in each cell's
`eval.json`. No re-run, no GPU.

| | recorded | corrected |
|---|---|---|
| n_qualifying | **2** | **4** |
| qualifying | e1_p075, e2_p050 | e1_p075, **e1_p100**, e2_p050, **e3_p025** |

Both flips came from loss noise: `e1_p100` recorded 0.0402 (below band) but averages 0.2437;
`e3_p025` recorded 0.7684 (above band) but averages 0.3254.

**The gap criterion did all the discriminating.** Nine of twelve cells sit inside the loss band on
averaged loss, so loss separated almost nothing; every pass/fail flip traces to single-example
noise. If the operating-point definition is to rest on one criterion, it should rest on gap.

The original record is **preserved unmodified** at
`gs://jenna-pdpo-pdpo/experiments/stage1_operating_point/GATE1_STATUS.json` (2026-08-18T11:33:32Z);
the correction is a separate artifact, `analysis_cpu/lossgap/GATE1_STATUS_CORRECTED.json`.
Note also that the *local* `experiments/stage1_operating_point/GATE1_STATUS.json` is a stale stub
from 2026-08-14 (`evaluated_checkpoints: 0`, `verdict: FAIL`) and is not the gate record.

**E9 model selection stands.** Both `pku|e1|r16|p075` and `pku|e2|r16|p050` qualify under
corrected loss, so the utility experiment's model choice needs no restatement — though its
rationale should read "gap in band, loss in band when averaged", and two further cells were
eligible that the recorded gate hid.

## AMENDMENT 2026-08-23 (B4) — CLAIM WORDING FOR THE LOSS-GAP RESULT

To be used verbatim; **do not** write that the result violates the Yeom bound.

> The generalization-error term that bounds loss-threshold membership attacks is **negative** here
> — at `rr_eps0.0|e2|r16|p100`, held-out true-label DPO loss 0.8164 minus train 1.8358 gives
> **−1.0194** — so it cannot account for a magnitude-based membership advantage of 0.957
> (AUC 0.9785). The attack lies **outside what the bound covers**; this is not a counterexample
> to the bound.

**Mechanism sentence, to appear alongside it:** the model memorises margin **magnitude**, not label
**direction**. RR randomises direction and leaves magnitude untouched. Therefore label-DP cannot
bound magnitude-based membership inference at any ε.

**Framing.** Lead with the as-trained column: RR train loss on the labels the model was actually
given falls **0.5981 → 0.0455**, against clean's **0.5395 → 0.0046**. RR does not reduce
memorisation; it randomises its direction. The negative loss gap corroborates that; it is not the
headline.

**BIMODALITY MUST BE STATED WHEREVER −1.02 APPEARS BESIDE SIGNED AUC 0.52.** Measured at
`e2|p100` (C1), members split by `rr_flipped` against one shared non-member pool:

| group | n | mean margin | 95% CI | frac > 0 | mean \|s\| |
|---|---:|---:|---|---:|---:|
| members UNFLIPPED | 1042 | **+7.5356** | [7.3661, 7.7101] | 0.9990 | 7.5358 |
| members FLIPPED | 958 | **−7.8005** | [−7.9812, −7.6186] | 0.0000 | 7.8005 |
| non-members | 2000 | −0.0210 | [−0.1105, 0.0681] | 0.5000 | 1.5575 |

Two near-symmetric member modes (|ratio| 1.0351) straddling a unimodal non-member distribution at
zero. Their weighted sum is **+0.1896** — which is all the signed statistic can see, hence AUC
0.5188. Magnitude sees 7.5–7.8 against 1.5575, hence AUC 0.9785. Observed mode weights
0.5210/0.4790 against the realized flip rate 0.5105/0.4895: binomial **z = −0.939, consistent at
95%**. The apparent contradiction between a −1.02 mean gap and a chance-level signed AUC is
resolved by the bimodality, and we state it rather than leaving it for a reviewer.

The 0.9990 / 0.0000 positive fractions are worth stating too: the model has learned its flipped
training labels essentially perfectly, while held-out sits at exactly 0.5000.

## AMENDMENT 2026-08-23 (C3) — OPEN, UNEXPLAINED: THE LENGTH-CORRELATION INVERSION

Correlation between held-out chosen-minus-rejected length asymmetry and the held-out margin,
eps=0 arm, same 3,000 rows throughout:

| steps | cell | r(Δlen, margin) | acc \| chosen longer | acc \| chosen shorter |
|---:|---|---:|---:|---:|
| 125 | e1_p025 | **+0.5827** | 0.6174 | 0.4051 |
| 250 | e1_p050 | +0.0616 | 0.4829 | 0.4856 |
| 375 | e1_p075 | +0.1368 | 0.5645 | 0.4661 |
| 500 | e1_p100 | **−0.3031** | 0.4509 | 0.5568 |
| 1000 | e2_p100 | −0.1599 | 0.4746 | 0.5356 |

At ε = 0 the labels are random with respect to length, so r should **decay toward 0**. It does not:
it inverts and stays negative. **Recorded as an open, unexplained observation.** No mechanism is
proposed and no claim rests on it.

Related but distinct, and not to be conflated: T6 established that on PKU length does **not**
predict membership (blind AUC 0.4934, CI straddling 0.5); E8 concerns length driving preference
*direction*. The inversion above is in E8's domain, not T6's.

---

## AMENDMENT 2026-08-23 (A4) — APPLY THE REALIZED RATE OF THE *SUBSAMPLE*, NOT THE FILE

A0.1 fixed the analysis axis to realized flip rate and tabulated it over all 8,000 rows. That
table is correct for the arm, but the MIA scores a deterministic 2,000-row **subsample** of the
members, and that draw has its own realized rate. At eps=0 the subsample contains 958 flipped of
2,000 -> **p_sub = 0.4790**, giving `1 - 2p = 0.0420`, versus the file-wide `0.02100`. The two are
statistically consistent (binomial z = -0.939) but differ by 2x, and predictions built on the
file-wide value are low by that factor.

**Rule: any statistic computed on the MIA subsample is predicted from the subsample's realized
rate.** The A0.1 table stands for arm-level description; it is not the right denominator for
subsample statistics. Both are to be reported.

## AMENDMENT 2026-08-23 (A4) — INTERNAL ATTENUATION REFERENCE, AND WHY IT WAS NEEDED

Fit-matching the eps=0 arm to the Stage 1 clean curve is **impossible**: all 12 cells have measured
gap in [-0.0077, 0.0395], below the clean curve's support floor of 0.1112. `report_rr_arm.py`
refuses to extrapolate, so all 12 report `out of support`. This was flagged before launch and is
now confirmed for the whole arm.

The flip split supplies an internal reference at identical fit. AUC is linear in the positive-class
mixture, so with members split by `rr_flipped` against one shared non-member pool:

    AUC_mix = w_u * AUC_u + w_f * AUC_f        -- verified to 1.11e-16 over 12 cells

and under antisymmetry (`AUC_f = 1 - AUC_u`) this collapses to the pre-registered form

    AUC_mix - 0.5 = (1 - 2p) * (AUC_u - 0.5)

with `AUC_u - 0.5` as the unattenuated separation. **Antisymmetry holds approximately, not
exactly**: residuals -0.0065 to +0.0335, mean +0.0067.

**Result.** For the eight cells with a large reference (`AUC_u >= 0.95`), obs/pred spans
**0.919-1.123**, mean **1.024**, and at `e3|p100` it is **0.997**. The `(1-2p)` attenuation form is
quantitatively confirmed, not merely confirmed in sign.

The four early-training cells (`AUC_u` 0.62-0.86) give ratios 1.44-2.87. This is denominator
instability, not counter-evidence: with `(1-2p) = 0.042` the predicted excess is 0.005-0.015, the
same scale as the +/-0.019 bootstrap half-width on the observed AUC. **Cells with a small
unattenuated reference are uninformative about the attenuation factor and must not be averaged in.**

Note `e3|p025` is simultaneously the worst antisymmetry residual (+0.0335) and the only cell of 12
whose signed AUC CI excludes 0.50 ([0.5118, 0.5507]). Probably the same fact; recorded, not
explained.

## AMENDMENT 2026-08-23 (A4b) — FLIPPED vs UNFLIPPED MAGNITUDE: NO DETECTABLE DIFFERENCE

The within-model comparison the deliverable was instrumented for, all 12 eps=0 cells, fit held
exactly constant (same checkpoint, same weights, same non-member pool):

- **11 of 12** mean-`|s|` difference CIs include zero.
- **12 of 12** per-group magnitude AUC CIs overlap.
- Sole exception on the mean: `e2|p100`, +0.2646 [+0.0303, +0.5092] — but its magnitude AUCs still
  overlap (0.9759 unflipped vs 0.9813 flipped), so it does not survive as a leakage difference.
  One marginal hit in twelve is consistent with multiplicity.

**Verdict: being flipped does not change how much a member leaks under the magnitude statistic.**
This is the correct, unconfounded test, and it replaces the cross-model comparison (eps=0 magnitude
0.6322 vs clean control 0.5900) that must NOT be used as evidence -- those two models differ in fit
(gap 0.027 vs 0.111) and, as recorded above, the eps=0 arm cannot be fit-matched to the clean curve
at all.

Magnitude memorisation is indifferent to label direction. That is the measured basis for the
mechanism sentence in the B4 amendment.

---

## AMENDMENT 2026-08-23 — STAGE 2 SPEND CEILING RAISED $90 -> $110

At the measured 2.8 h/cell, the 48 cells remaining after the eps=0 arm project to ~$76 more
(~$98 total). The $90 ceiling would therefore have halted the sweep around cell 55 of 60 and
**truncated the eps=2.0 arm** -- the arm where the realized and nominal attenuation axes diverge
most (0.78300 vs 0.76159) and so the one least safe to lose. $20 of headroom is cheaper than
another stop-fix-verify cycle.

Eval batching is explicitly **NOT** being attempted. The sweep is not to be stopped again for
optimization; fix (a) shipped and that is the end of the optimization work for this stage.

Ceiling enforced in `experiments/stage2_health.py` (`SPEND_CEILING_USD`), deployed to the running
supervisor in place. The halt path is unchanged: delete the GPU VM, then write `verdict: STOPPED`.
