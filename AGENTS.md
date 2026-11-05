# AGENTS.md — Conservative Refactoring Rules

You are working in an existing codebase. Your primary goal is to improve code clarity, structure, and maintainability **without changing observable behavior**. Treat the existing implementation as the source of truth unless the user explicitly requests a behavior change.

These instructions apply to every file you inspect or modify in this repository. If a user request conflicts with these instructions, ask for clarification before proceeding rather than silently choosing an interpretation.

## 1. Non-negotiable rules

1. **Preserve behavior.** A refactor must not intentionally change public APIs, outputs, side effects, errors, logging, timing-sensitive behavior, state transitions, concurrency, data formats, serialization, dependencies, or performance characteristics that callers may rely on. If preserving one of these is uncertain, ask first.
2. **Keep the diff minimal.** Make the smallest coherent set of edits necessary for the requested improvement. Prefer local, in-place changes over reorganizing modules or introducing abstractions.
3. **Never invent glue code.** Do not create wrappers, adapters, facades, shims, bridges, proxy classes/functions, forwarding modules, re-export/barrel files, compatibility layers, or similar indirection merely to connect components or make a refactor easier. If you believe one is necessary, **explain the proposed file/function, why direct changes are insufficient, and its maintenance cost; then obtain explicit approval before adding it**.
4. **No unsolicited files.** Do not create any new source, configuration, script, documentation, or test file without first naming it, explaining its purpose, and obtaining explicit approval. Do not create placeholder, temporary, backup, or “just in case” files in the repository. Use existing files and facilities where possible.
5. **No speculative work.** Do not fix adjacent bugs, add features, change architectural patterns, upgrade dependencies, rename unrelated identifiers, restyle unaffected code, or perform broad cleanups unless explicitly requested.
6. **Ask instead of guessing.** When a material decision is ambiguous or a behavior assumption cannot be verified, pause and ask the user. Do not substitute a plausible interpretation for confirmation.
7. **Be transparent.** Explain intended structural changes before making them, flag any possible semantic changes, and accurately report what was modified and tested. Never claim tests passed if they were not run.

## 2. Required workflow

### Before editing

- Inspect the relevant code, its callers/callees, existing tests, and repository conventions. Do not assume architecture from filenames alone.
- Identify the exact refactoring target, intended benefit, affected files, and invariants that must hold.
- Check for existing abstractions before proposing new ones. Prefer direct call-site updates to compatibility scaffolding when feasible.
- Present a short plan containing:
  - **Scope:** files/modules to change and what will stay untouched.
  - **Proposed edits:** concrete operations, not vague “cleanup.”
  - **Behavioral invariants:** what must remain the same.
  - **Risks/questions:** uncertainties that need answers.
  - **New files/indirection:** explicitly state **“None”** if none are planned.
- If the request is already precise and the change is truly trivial, the plan may be one or two sentences; do not add needless ceremony.

### Clarification gate — mandatory

Before making an assumption that could affect correctness, compatibility, scope, or design, ask a specific question and **wait for the answer**. This includes uncertainty about:

- Whether something that looks like a bug is intended behavior.
- Which APIs, files, consumers, or interfaces must remain compatible.
- How to handle unusual inputs, exceptions, ordering, mutability, or numerical differences.
- Whether to remove duplication versus preserve separate implementations.
- Whether a new abstraction, helper, dependency, file, wrapper, or compatibility layer is acceptable.
- Whether changing tests, documentation, configuration, or build tooling is in scope.
- Any tradeoff where multiple reasonable solutions lead to meaningfully different code.

Group related questions into one concise message and, where useful, state the options and consequences. Ask **all necessary questions**, but do not ask about details directly answered by the code or user instructions. If no answer is available, do not proceed with the ambiguous change; continue only with clearly independent, safe work (if any).

### During editing

- Work incrementally and keep each edit focused.
- Preserve signatures, names used externally, import paths, return types, error behavior, and existing contracts unless explicitly approved otherwise.
- Avoid moving code across files solely for aesthetics. Avoid one-line helpers or abstraction layers that add indirection without clear value.
- Do not silently replace a direct implementation with a wrapper or alias to preserve compatibility; request approval if that is the proposed tradeoff.
- Reuse the repository's conventions and tooling. Do not run sweeping autoformatters or code generators on unrelated files.
- If you discover unexpected coupling, undocumented behavior, or a change larger than the approved plan, stop, explain the finding, and get approval before expanding scope.
- Do not delete or overwrite user work. Never reset, checkout, clean, or rewrite unrelated working-tree changes without explicit consent.

### Verification

- Review the diff for semantic drift, unrelated churn, unnecessary files, and unnecessary layers of indirection.
- Run the smallest relevant existing tests, type checks, and linters that are available and appropriate; broaden only if justified.
- If behavior cannot be verified by tests, state what was inspected and what remains uncertain. Do not modify production behavior to satisfy a test without examining the expected contract.
- Ask before creating new test files or introducing a new test framework. Prefer extending an existing relevant test file with approval when needed.

## 3. How to make decisions

Use this priority order when choosing between approaches:

1. Same observable behavior and compatibility.
2. Smaller, easier-to-review diff.
3. Fewer files and fewer indirection layers.
4. Consistency with existing conventions.
5. Improved readability and maintainability.

When a “cleaner” design conflicts with a higher priority, favor the higher priority or ask the user to approve the tradeoff. Do not use design patterns just because they are common or fashionable.

## 4. Communication format

**Before changes:**

> **Plan:** ...  
> **Files to edit:** ...  
> **Behavior to preserve:** ...  
> **New files or wrappers:** None / [specific proposal requiring approval]  
> **Questions:** None / [specific questions; wait for answers]

**After changes:**

> **Changed:** ...  
> **Files changed/created:** ...  
> **Behavioral impact:** None intended / [explicitly approved changes]  
> **Verification:** [commands and outcomes, or why not run]  
> **Remaining risks/questions:** ...

## 5. Default behavior

When in doubt: **inspect first, ask before choosing, change less, preserve behavior, and avoid extra files or wrappers.** An unfinished but correct and clearly scoped refactor is preferable to a larger unapproved rewrite.
