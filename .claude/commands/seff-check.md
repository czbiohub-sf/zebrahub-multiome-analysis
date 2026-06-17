---
description: Audit a completed SLURM job's resource usage with seff and record it (enforces the workspace CLAUDE.md mandate)
---

Audit SLURM job resource usage. Usage: `/seff-check <jobid> [job-type]`
Arguments: $ARGUMENTS

This command ENFORCES the workspace-level mandate in
`/hpc/projects/data.science/yangjoon.kim/CLAUDE.md` (SLURM Resource Monitoring
Workflow) — it is not a new optional process; it codifies a required one.

Steps:

1. Run `seff <jobid>` and parse: requested vs. actual CPUs, memory, walltime,
   and the CPU / memory efficiency percentages.

2. Flag against these thresholds:
   - Memory efficiency < 50% → recommend a lower `--mem`
   - CPU efficiency < 50% → recommend fewer `--cpus-per-task`
   - Walltime used < 25% of requested → recommend a shorter `--time`
   - Exit due to OOM → note the minimum memory needed

3. Record the finding in the memory file `slurm-resources.md` (per the
   workspace CLAUDE.md). Use a consistent row:
   `| <date> | <job-type> | <jobid> | req: CPUs/mem/time | actual: CPU%/mem/walltime | recommendation |`

4. If `slurm-resources.md` already has an entry for the same job-type, show the
   delta vs. the last run and whether prior recommendations were applied.

5. Give a one-line recommended `sbatch` resource line for the next similar job.

Keep it concise — this runs after every SLURM job, so the output should be a
quick scannable audit, not an essay.
