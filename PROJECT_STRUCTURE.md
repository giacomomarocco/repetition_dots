# Project structure

## Current layout

The repository separates reusable experiment logic from executable workflows:

```text
filler/
  addition/    one-fact, two-fact, modulo-10, and accuracy-analysis logic
  fact_eval/   shared fact protocol plus Hugging Face and SGLang backends
  dsv4/        DeepSeek V4 Logit Lens, factorial, and hook primitives
  qwen.py      local Qwen model and prompt utilities
scripts/
  addition/    thin addition command wrappers
  fact_eval/   thin fact-evaluation command wrappers
  dsv4/        preparation, capture, analysis, plotting, and validation commands
tests/         tests of package and workflow behavior
notebooks/     interactive analyses
runs/          generated experiment outputs and checkpoints
model/         local model weights and configuration
ports/         pinned third-party inference ports
```

Each package directory has an `__init__.py` to make the package boundary
explicit and predictable to Python, pytest, IDEs, and static-analysis tools.
The files do not re-export implementation details; imports should name the
owning module directly.

Run workflows from the repository root with `python -m`, for example:

```bash
.venv/bin/python -m scripts.run_qwen
.venv-sglang/bin/python -m scripts.addition.two_fact --help
.venv-sglang/bin/python -m scripts.dsv4.validate_logit_lens --help
```

Code used by more than one command belongs in `filler/`. Files in `scripts/`
should contain command orchestration or a small wrapper around a package
`main()` function. New experiment domains should use matching subdirectories in
both trees when they need reusable logic and commands.

## 2026-09-03 reorganization

The project previously kept 27 unrelated Python files at repository root. They
were grouped by responsibility under `filler/` and `scripts/`; internal imports,
tests, shell launchers, notebook imports, and documented commands were updated
at the same time. The third-party `ports/` trees and generated data were not
altered.

Important module moves include:

- `one_fact_addition_sglang.py` to `filler/addition/one_fact.py`, invoked through
  `scripts.addition.one_fact`;
- `evaluate_facts_sglang.py` to `filler/fact_eval/sglang.py`, invoked through
  `scripts.fact_eval.sglang`;
- `dsv4_factorial.py`, `deepseek_v4_logit_lens.py`, and `dsv4_lens_hooks.py` to
  `filler/dsv4/factorial.py`, `lens.py`, and `hooks.py`;
- operational DeepSeek utilities to `scripts/dsv4/` with shorter names because
  their directory now supplies the domain context.

This was a source-layout change only. Existing run outputs, checkpoints, model
assets, and experiment parameters were preserved.
