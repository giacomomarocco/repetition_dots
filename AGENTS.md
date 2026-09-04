# Project Agent Guidance

## Expensive model-launch preflight

Before starting an expensive model load or warmup, inspect the workspace for
existing task-specific integration and define the end-to-end validation first.
Confirm every launch-time-only requirement in the actual command, including
activation hooks, hidden-state flags, tensor dumps, and native readout logic.

Do not warm an uninstrumented server for an activation-analysis task. For
DeepSeek V4 Logit Lens or intervention work, inspect and use
`filler/dsv4/factorial.py` (`NativeResidualHooks` and native mHC projection), run its
import/unit checks, and define the final-layer equivalence test before launch.
If a warmed process cannot expose required experiment state, report that
immediately instead of preserving it solely because it is warm.

## Project structure

- `model/` contains locally downloaded model weights and their associated configuration and tokenizer files.
- `ports/` contains third-party inference ports and pinned upstream source checkouts used by those ports.
- `runs/` contains experiment outputs, checkpoints, logs, and run configuration manifests.
- `filler/` is the importable Python package for reusable experiment logic.
- `scripts/` contains thin, domain-grouped executable workflows; invoke them with
  `python -m scripts.<domain>.<command>` from the repository root.

Keep this section current whenever making high-level structural changes to the project, such as adding, removing, renaming, or repurposing top-level directories.

## Project lab log

Treat documentation of substantive analyses and experiments as a project lab
log. Before ending such work, create or update a durable Markdown record in the
workspace. Prefer a focused existing document when one covers the work; create
a clearly named record when none does. Use dated, chronological entries when a
document spans multiple sessions.

Record enough context for a later agent or collaborator to understand and
reproduce the work without relying on chat history. Include, as applicable:

- the date, objective, and motivation or research question;
- inputs, datasets, model/software versions, and relevant revisions;
- the method, important commands, parameters, and execution environment;
- observations, intermediate findings, and decisions with their rationale;
- final results, validation or integrity checks, and output locations;
- failures, fixes, limitations, unresolved questions, and useful next steps.

Keep entries concise but preserve the facts and reasoning that maintain project
continuity. Link to large generated artifacts instead of copying them into the
log. Never record credentials, tokens, private keys, OTPs, or other secrets.

## Version control

Organize completed work into small, focused commits with descriptive messages.
Before committing, present the working-tree changes for user inspection and wait
for approval; do not combine unrelated user changes into the same commit.
