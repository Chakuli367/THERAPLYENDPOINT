# Theraply — Disorder Exercise Map

Planning doc only — no code yet. Goal: list every exercise that could fill the Part 4 (interactive exercise) slot within a session, per disorder cluster, and propose how to split this across files once we start building.

---

## 1. Proposed file structure

Right now everything lives in `therapy_app.py`: one `PROGRAM` list, one `EXERCISE_CATALOG` dict. To support multiple disorder tracks without that file ballooning further:

```
programs/
  social_anxiety.py        # existing PROGRAM, moved as-is
  gad.py                   # new 8-week GAD program
  panic_health_anxiety.py  # new 8-week program
  ocd_intrusive.py         # new 8-week program
  insomnia_module.py       # 2–4 week module, not a full track — gets spliced into any active track

exercises/
  shared_catalog.py        # existing entries reusable across tracks (breathing, grounding, body_scan,
                            # thought_record, fear_ladder, exposure_commitment, exposure_debrief,
                            # safety_behaviors_audit, coping_cards, values_compass, progress_story,
                            # anxiety_mapping, probability_pie, evidence_court, prediction_logger)
  gad_catalog.py            # GAD-specific new exercise types
  panic_health_anxiety_catalog.py
  ocd_intrusive_catalog.py
  insomnia_catalog.py

tracks.py                  # TRACKS = {track_id: program_list}; get_program_session(track_id, week, session)
                            # get_exercise_catalog(track_id) merges shared_catalog + that track's catalog
screening.py                # intake questionnaire → track_id + active_modules + safety triage
```

`therapy_app.py` keeps the Flask routes but imports from these instead of defining `PROGRAM` / `EXERCISE_CATALOG` inline. This is a mechanical split — nothing about today's social anxiety track changes.

---

## 2. Shared pool — already built, just needs `best_for` tags widened

These need zero new code, only tag updates so they surface correctly across tracks:

| Exercise | Reused for |
|---|---|
| `breathing_box`, `breathing_4_7_8` | GAD, panic, insomnia |
| `grounding_5_4_3_2_1` | panic, OCD (dissociation during high distress) |
| `body_scan` | GAD, panic, OCD |
| `thought_record` | GAD, health anxiety |
| `evidence_court` | GAD, health anxiety |
| `probability_pie` | GAD, panic, health anxiety |
| `prediction_logger` | GAD, panic |
| `fear_ladder`, `exposure_commitment`, `exposure_debrief` | panic, health anxiety, OCD (renamed conceptually to "exposure," not just social) |
| `safety_behaviors_audit` | panic, health anxiety, OCD (relabel as compulsions/checking where needed) |
| `coping_cards` | every track |
| `values_compass`, `progress_story` | every track |
| `anxiety_mapping` | GAD, panic |

---

## 3. Generalized Anxiety & Worry — exercises needed

**New builds (5):**
- `worry_tree` — branch: is this a solvable problem (act on it) or a hypothetical "what if" (practice letting go)?
- `meta_worry_record` — captures beliefs about worrying itself ("worrying keeps me prepared"), not the worry content.
- `worry_time_scheduler` — fixed daily worry window; intrusive worries outside it get logged and deferred there.
- `uncertainty_log` — deliberately leaves something unresolved, tracks anxiety over time without seeking reassurance. *(shared with OCD track)*
- `progressive_muscle_relaxation` — tense-release through muscle groups for chronic somatic tension. *(shared with insomnia module)*

**Reused from shared pool:** anxiety_mapping, body_scan, thought_record, evidence_court, probability_pie, prediction_logger, fear_ladder, coping_cards, safety_behaviors_audit, values_compass, progress_story.

That's roughly 16 slots (8 weeks × 2 sessions) covered with 5 new + 11 reused.

---

## 4. Panic Disorder & Health Anxiety — exercises needed

**New builds (5):**
- `panic_cycle_mapper` — interactive sensation → catastrophic thought → fear → more-sensation loop, personalized.
- `symptom_interpretation_record` — thought-record variant where the trigger is always a physical sensation: felt sensation → medical catastrophe → alternative benign explanation.
- `checking_behavior_log` — tracks health-anxiety compulsions (pulse-checking, symptom-googling, body-scanning for lumps).
- `reassurance_fasting_commitment` — commits to resisting reassurance-seeking (doctor visits, asking "do I look okay") for a set period.
- `interoceptive_exposure` — deliberately induces panic-like sensations (spinning, breath-holding, stair climbing) so the brain learns they're not dangerous. **Flag: needs a pre-exercise screening gate** (cardiac/respiratory conditions, pregnancy) before this ships — it's the one exercise in this whole map I'd want extra review on.

**Reused from shared pool:** anxiety_mapping, body_scan, breathing_box/4_7_8, probability_pie, evidence_court, exposure_commitment, exposure_debrief, safety_behaviors_audit, coping_cards, values_compass, progress_story.

---

## 5. OCD & Intrusive Thoughts — exercises needed

Important framing note before the list: your existing `evidence_court` / `thought_record` pattern (weigh evidence for/against a thought) is the wrong tool for obsessions — debating whether an intrusive thought is "true" functions as reassurance-seeking, which reinforces the OCD loop instead of breaking it. The exercises below are built around accepting uncertainty, not resolving it.

**New builds (7):**
- `obsession_domain_mapping` — anxiety_mapping's structure, but domains are checking, contamination, symmetry, harm, religious/moral, relationship, etc.
- `intrusive_thought_normalizing` — psychoeducation-as-exercise distinguishing "having a thought" from "the thought meaning something about you" (thought-action fusion).
- `compulsion_log` — obsession trigger, intrusive thought, compulsion performed, urge intensity before/after, time spent.
- `erp_hierarchy_builder` — fear_ladder's structure, ranking exposures by anticipated distress, each rung naming which compulsion must be resisted.
- `response_prevention_commitment` — exposure_commitment's structure plus an explicit "compulsion I will not perform" field.
- `urge_surfing_timer` — timed exercise (like breathing_box's UI) for riding out a compulsion urge with check-ins, built around "the urge peaks and passes."
- `reassurance_resistance_log` — tracks resisting asking others for reassurance specifically (distinct from checking_behavior_log, which is solo checking behaviors).

**Reused/shared:** body_scan, exposure_debrief (relabeled for ERP), safety_behaviors_audit (relabeled as mental rituals), uncertainty_log *(shared with GAD)*, coping_cards, values_compass, progress_story.

**Screening note:** gate entry to this track behind an intake question. If responses suggest harm-OCD content involving risk to self or others, route straight to crisis resources instead of into the exercise flow — that's triage, not a CBT exercise.

---

## 6. Insomnia Module (2–4 weeks, layered onto any track)

**New builds (4):**
- `sleep_diary` — bedtime, latency to sleep, wake-ups, total sleep, subjective quality. Feeds the next exercise.
- `sleep_restriction_planner` — computes a sleep window from diary data (sleep-efficiency calc). **Flag: needs a screening gate** — contraindicated for seizure disorders, can trigger mania in bipolar disorder.
- `stimulus_control_checklist` — adherence to bed-is-for-sleep rules (no screens, get up after 20 min awake, fixed wake time).
- `sleep_belief_record` — thought_record adapted to sleep-specific beliefs ("I need 8 hours or tomorrow is ruined").

**Reused/shared:** progressive_muscle_relaxation *(from GAD)*, worry_time_scheduler *(from GAD, for pre-sleep worry spirals)*.

---

## 7. Tally

| Track | New exercise types | Reused | Notes |
|---|---|---|---|
| GAD | 5 | 11 | Closest to existing architecture — good first build |
| Panic / Health Anxiety | 5 | 11 | One gated exercise (interoceptive_exposure) |
| OCD / Intrusive Thoughts | 7 | 7 | Needs screening gate + reframed evidence-based pattern |
| Insomnia (module) | 4 | 2 | One gated exercise (sleep_restriction_planner); not a full 8-week track |

**21 new exercise types total**, 3 of them carrying a screening/safety gate, 2 shared across multiple tracks (`uncertainty_log`, `progressive_muscle_relaxation`/`worry_time_scheduler`).

---

## Next steps (pick when ready)

1. Week-by-week session breakdown for one track (psychoeducation text + which exercise lands where) — same format as the existing `PROGRAM`.
2. Screening questionnaire design (`screening.py`) — questions, scoring, track routing, safety short-circuit.
3. Actual `EXERCISE_CATALOG`-format config for any of the 21 new exercise types.
