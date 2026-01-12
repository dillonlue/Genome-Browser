# shapley_masking
# Collaboration & Workflow Guidelines

This repository follows a simple, professional GitHub workflow designed to **maximize collaboration without losing code understanding or ownership**.

These guidelines are optimized for **small teams (1–2 contributors)** and scale naturally as the project grows.

---

## Goals

* Maintain clear ownership of the codebase
* Ensure all changes are understandable and reviewable
* Avoid large, confusing commits
* Enable collaboration without sacrificing quality

---

## Branching Model

### `main` branch

* The `main` branch is **protected**
* No direct commits are allowed
* All changes must go through Pull Requests (PRs)

### Feature branches

* All work happens on short-lived feature branches
* Branch naming should be descriptive, e.g.:

  * `feature/auth-flow`
  * `fix/login-bug`
  * `docs/api-readme`

---

## Pull Request Process

### Required rules

* Every change must be submitted as a Pull Request
* At least **one reviewer is required** (usually the other collaborator)
* PRs should be **small and focused**

### PR expectations

Each Pull Request should clearly explain:

1. **What problem is being solved**
2. **High-level approach** (how the solution works)
3. **Any risks or trade-offs**

Large or unfocused PRs may be rejected and split into smaller ones.

---

## Review Philosophy

Reviews focus on **architecture and clarity**, not personal style preferences.

Reviewers should look for:

* Alignment with existing abstractions
* Clear responsibilities and boundaries
* Avoidance of unnecessary complexity
* Long-term maintainability

If a change is hard to understand, it needs revision — even if it works.

---

## Code Ownership

* The repository owner is the **final gatekeeper** for changes
* Contributors are encouraged to commit freely on their own branches
* Integration into `main` only happens after review

As trust grows, merge permissions may be expanded per subsystem or directory.

---

## Documentation Standards

Any change that:

* Introduces a new abstraction
* Alters control flow
* Adds non-obvious behavior

Must include one or more of:

* Inline comments explaining *why*, not *what*
* Updates to `README.md`
* Notes in `docs/`

Documentation is how understanding persists over time.

---

## When Direct Commits Are Acceptable

Direct commits to `main` are discouraged.

They may be acceptable **only** for:

* Trivial documentation changes
* Typos or formatting fixes
* Extremely low-risk updates

Even then, changes should be reviewed promptly and reverted if necessary.

---

## Summary

This workflow prioritizes:

* Code comprehension
* High signal-to-noise reviews
* Sustainable collaboration

The goal is not process for its own sake, but a codebase that **remains understandable months and years later**.

---

If you’re contributing to this repo, thank you for keeping these standards high 🙌
