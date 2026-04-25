---
name: start-task
description: Use this at the start of every coding task, before making any changes. Checks out main, pulls latest, creates a new branch, and sets up for a PR.
---

Before writing any code, follow this git workflow:

1. Run `git checkout main` to switch to the main branch.
2. Run `git pull origin main` to get the latest changes.
3. Ask the user for a branch name if they haven't provided one, or infer a short descriptive name from the task. Use the format `<username>/<short-description>` where `<username>` is the person requesting the work (not "claude") — default to the output of `git config user.name` if not specified (e.g. `plasorak/fix-broken-flush`).
4. Run `git checkout -b <branch-name>` to create and switch to the new branch.

Then proceed with the coding task. When the work is done:

5. Commit the changes with a clear message explaining *why* the change was made, not just what changed. Always add the co-author trailer:
   ```
   Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
   ```
6. Push the branch with `git push -u origin <branch-name>`.
7. Open a PR against `main` using `gh pr create --base main` with a short title and a summary covering what changed and why.

Return the PR URL to the user when done.

When you see changes that are uncommitted that you didn't do, ask the user what they want to do with it. It's likely they made manual changes and will want the changes in the PR.