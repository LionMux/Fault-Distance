# Integration Status — Variant A (~100 ML papers notebook)

## Phase 0 — Framework & master notebook
- [x] Create Variant A master workflow + template framework (`deep_dive_notebook/framework.md`)
- [x] Create initial master notebook skeleton (`deep_dive_notebook/VARIANT_A_deep_dive_study_notebook.md`)

## Phase 1 — Candidate discovery (per-domain)
- [x] Fault candidates created (`deep_dive_notebook/papers/fault/paper_candidates.jsonl`)
- [x] Forecast candidates created (`deep_dive_notebook/papers/forecast/paper_candidates.jsonl`)
- [x] Stability candidates created (initial attempt had placeholders; **fixed**)
  - [x] Regenerate/replace stability candidates with verified arXiv IDs (current file now contains real arXiv entries for many items)
  - [ ] Remaining: ensure *all* 20 stability entries have fully verified bibliographic fidelity (authors/citation correctness; no structural `null` fields remain)
- [x] Security candidates created (`deep_dive_notebook/papers/security/paper_candidates.jsonl`)

## Phase 2 — Relevance filtering + exact 100 selection
- [ ] Dedupe candidates across domains
- [ ] Strict relevance filter: “scientific + popular + relevant to ML training/application in power/energy sector”
- [ ] Select exactly 100 papers and write `deep_dive_notebook/papers/selected_candidates.jsonl`

## Phase 3 — Authoring: per-paper deep-dive skeletons
- [ ] Generate 100 per-paper skeleton markdown files using the framework template:
  - `deep_dive_notebook/per_paper/P001.md ... P100.md`
- [ ] Create mapping `paper_to_template_map.jsonl`

## Phase 4 — Compile final master notebook
- [ ] Insert all 100 deep-dive skeletons into:
  - `deep_dive_notebook/VARIANT_A_deep_dive_study_notebook.md`

## Phase 5 — Quality pass
- [ ] Check citation integrity (no fabricated placeholders; all IDs verified)
- [ ] Check mathematical rigor template completeness
- [ ] Check physical meaning mapping for metrics
