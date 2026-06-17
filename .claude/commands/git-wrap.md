---
description: Wrap up a work session with clean git hygiene (audit → group commits → PR → cleanup)
---

Help wrap up this coding session with clean git hygiene. Optional context from the user: $ARGUMENTS

Follow these steps, confirming with me before any irreversible or outward-facing action:

1. **Audit.** Run `git branch --show-current`, `git status --short`, and
   `git log --oneline main..HEAD` (commits not yet on main) plus
   `git log --oneline origin/main..HEAD`. Distinguish:
   - commits already pushed/merged,
   - local commits not pushed,
   - uncommitted working-tree changes (and whether they're from THIS session
     or pre-existing leftovers — do not assume they're mine).

2. **Group.** Propose logical commit groupings by theme (don't sweep unrelated
   changes into one commit). Show me the groupings and proposed commit messages.
   Wait for my confirmation before committing.

3. **Branch discipline.** If on `main` (default branch), propose creating a
   feature branch first unless I explicitly say commit-to-main. Never push to
   the default branch without explicit confirmation.

4. **Commit + push.** After I confirm, commit in the agreed groups and push the
   branch.

5. **PR.** Draft a PR title + concise body (summary + scope). Show me before
   opening. End the body with the Claude Code attribution footer.

6. **Cleanup + report.** After merge (when I confirm), update local main
   (`git fetch && git merge --ff-only origin/main`), delete merged branches,
   and report any leftover untracked files that may need `.gitignore` entries.
   Verify `git status` is clean at the end.

Reminders:
- `git stash` is the reversible move when a branch switch is blocked by tracked
  edits whose intent is unclear.
- `.gitignore` never deletes on-disk files — after adding ignore rules for
  generated dirs, confirm with `ls` that the files remain.
- Tracked files under a gitignored dir can't be re-`git add`ed without `-f` —
  honor the convention (e.g. `outputs/` = build artifacts) rather than forcing.
