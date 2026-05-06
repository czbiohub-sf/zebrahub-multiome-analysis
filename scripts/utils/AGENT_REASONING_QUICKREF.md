# Agent Reasoning — Quick Reference

A 1-page companion to [`AGENT_REASONING_LESSONS.md`](./AGENT_REASONING_LESSONS.md).
Each principle is one line; the long-form doc has the concrete examples
and the operational rule.

---

## The 14 principles

```
At analysis start
  ① Re-ask: "what metric answers what question?"
     (specificity ≠ activity ≠ synthesizability)
  ③ Don't hard-code tissue/context biology; externalize via CLI args
     or agent-driven lookups
  ⑥ Bonuses for context-match, not penalties for "off-target"
     (multi-tissue genes are opportunities)

During the algorithm
  ④ Layer evidence; don't trust a single signal
     (ATAC ∩ RNA → validated)
  ⑤ Real-world constraints shape the algorithm, not just filter
     (IDT 500 bp cap → continuous synthesis_length_factor)
  ⑦ Make scoring transparent; preserve every component
  ⑧ Defaults for the 80% case + documented opt-out

At edge cases
  ② Verify annotations against the QUESTION, not the database default
     (the −10.7 kb "exonic" peak is in CT025909.4, not pax2a)
  ⑩ Iterate on edge cases; don't paper over them
     (36 bp degenerate stitching → iterative gap_threshold)

At each output step
  ⑨ State caveats next to the output (not on request)
  ⑭ Communicate decisions, not just results (one-line "why" per filter)

At completion
  ⑪ Build for the next gene, not just this one
     (gene_locus_explore.py orchestrator, sanity-tested on sox10)
  ⑫ Cleanup as part of the workflow (32 MB → 20 MB, 32 → 3 entries)
  ⑬ Ask the user when reversibility is at stake
     (auto-mode ≠ delete-mode)
```

---

## Where the principles live

| Form | Path | Audience |
|---|---|---|
| Long-form with examples | `scripts/utils/AGENT_REASONING_LESSONS.md` | Humans reading the repo / new contributors |
| Condensed, this file | `scripts/utils/AGENT_REASONING_QUICKREF.md` | At-a-glance reference |
| User memory | `~/.claude/.../memory/feedback_agent_reasoning_principles.md` | Future Claude Code sessions (auto-loaded) |
| Index entry | `MEMORY.md` | Triggers loading the principle file when relevant |
| AgenticCRE integration | `agenticCRE/.../agent-reasoning-integration-plan.md` | The downstream agent that runs ISM / synthesis design |

---

## Why these matter

Each principle maps to a **specific failure mode** caught and corrected
mid-session. Without them, the analysis would have shipped:

- **without ①**: locus-view figure with z-score height (misleading — z-score and accessibility are r=−0.12)
- **without ②**: a synthesis candidate that's actually CT025909.4's exon (wrong gene)
- **without ⑥**: optic_cup peaks penalized as "off-target" (they're alternative pax2a-driven tissues)
- **without ⑩**: 36 bp degenerate "stitched" designs as real results

These aren't abstractions — they're the actual mistakes we caught and
corrected during the pax2a → MHB synthetic-enhancer session, distilled
into rules so the next analysis doesn't re-discover them by trial and
error.

---

## When to consult which principle

| You are about to… | Consult |
|---|---|
| Choose a metric for ranking | ①, ⑦ |
| Hard-code a tissue/celltype list | ③, ⑥ |
| Apply a categorical label to data | ②, ⑨ |
| Add a filter / cutoff | ⑤, ⑦, ⑭ |
| Pick a default value | ⑧, ⑭ |
| Investigate an anomalous result | ②, ⑩ |
| Produce a figure or table | ⑨, ⑭ |
| Conclude an analysis | ⑪, ⑫ |
| Delete or overwrite something | ⑬ |
