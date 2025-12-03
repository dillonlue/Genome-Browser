# Command: #git-save

Purpose: Create a concise, descriptive commit for all current changes.

Steps:
1) Ensure you are in repo root; inspect changes with `git status` and `git diff --stat`.
2) Stage everything: `git add -A`.
3) Craft a clear commit message that summarizes what changed and why; if details are missing, ask for clarification before committing.
4) Commit: `git commit -m "<message>"`.
5) Verify clean tree: `git status` (should report nothing to commit).
