# Filler-Token Computation in DeepSeek V4 Flash

## Project summary

This project asks what computation DeepSeek V4 Flash performs when meaningless filler tokens improve accuracy on one-fact addition problems.

The central aim is not merely to locate information in the residual stream, but to distinguish among competing algorithms:

1. **Serial latent computation:** filler positions form a recurrent relay, with later positions consuming intermediate state written by earlier positions.
2. **Parallel computation:** multiple filler positions independently reread the problem and perform redundant or complementary computation before the answer position aggregates their results.
3. **Compute-and-broadcast:** one position computes a useful result, which is copied or broadcast across later filler positions.
4. **Hybrid computation:** the model combines serial propagation, repeated rereading, and aggregation.

The initial task is one-fact addition, for example a prompt containing a learned fact or variable value $A$, followed by an addition request $A+X$, filler dots, and then the answer. A two-fact task is a natural extension because it can reveal genuine parallel decomposition rather than merely redundant attempts at one calculation.

## Main research question

**How do filler positions causally transform and transmit the representations of $A$, $X$, and $A+X$, and does the resulting computation rely on serial communication between filler positions or parallel access to the original question?**

The desired evidential ladder is:

1. **Representation:** identify where $A$, $X$, and $A+X$ become decodable.
2. **Mediation:** show that intervening on those representations changes the answer.
3. **Variable-level causality:** transplant one input variable while holding the other fixed and obtain the predicted hybrid answer.
4. **Algorithm:** use path-restricted interventions to distinguish serial from parallel computation.

Attention patterns or lens plots alone are insufficient for the final claim.

## Starting point and relation to prior work

The main starting point is *Reading Between the Dots*, which studies filler-token reasoning in models including DeepSeek V3 and Kimi. Relevant existing analyses include:

- attention from the question to filler positions and from fillers to the answer;
- logit-lens-style localization;
- post-prefill key/value cache swaps;
- position-resolved interventions in multi-fact settings;
- activation patching and attention knockout experiments present in the public repository.

An important detail of the reported one-fact cache-swap experiment is that donor and target prompts were both fully prefilled. The intervention swapped filler-position keys and values at every layer and reran only the final generation-prompt token. It therefore did **not** transplant a state and allow its downstream filler descendants to recompute. When the donor and target shared $X$ but differed in $A$, the donor answer rose substantially in rank, but this remains compatible with several explanations: transplanting $A$, transplanting the already-computed answer, or moving a correlated feature.

The proposed work is new insofar as it uses factorial, variable-specific activation interchange and path-controlled recomputation to identify what a filler representation means and how it is used. Basic patching by itself is not the novelty.

## Can ordinary logit lens be used?

Yes. Ordinary logit lens is a good first-pass localizer, provided it is adapted to DeepSeek V4 Flash’s native hidden-state structure rather than applied to an arbitrary average of streams.

For a native hidden state $H_{\ell,p}$ at layer $\ell$ and position $p$, first apply the model’s hidden-state-combination operation, then the final normalization and unembedding:

\[
\operatorname{LL}_{\ell,p}(v)
=
\left[
W_U\,\operatorname{RMSNorm}
\left(
\operatorname{hc\_head}(H_{\ell,p})
\right)
\right]_v.
\]

Practical recommendations:

- start with a filler token that is represented as one token, such as a dot if tokenization permits;
- use single-token numerals initially;
- score a preregistered set of candidate values rather than relying only on top-token inspection;
- plot both raw logits and logits centered over the candidate set;
- separate examples used to discover layers and positions from examples used to test causal hypotheses;
- patch the complete native hidden state at a site, preserving all streams;
- use J-Lens later as a robustness check if ordinary logit lens misses information that is present but not yet aligned with the unembedding.

Logit lens can answer “where is this value readable?” It cannot by itself establish that the model uses that value or distinguish serial from parallel computation.

## Core factorial interchange experiment

Construct a $2\times2$ family of matched prompts:

| Prompt | Fact value | Addend | Correct sum |
|---|---:|---:|---:|
| $P_{00}$ | $A_0$ | $X_0$ | $S_{00}=A_0+X_0$ |
| $P_{10}$ | $A_1$ | $X_0$ | $S_{10}=A_1+X_0$ |
| $P_{01}$ | $A_0$ | $X_1$ | $S_{01}=A_0+X_1$ |
| $P_{11}$ | $A_1$ | $X_1$ | $S_{11}=A_1+X_1$ |

Choose values so all four sums are distinct and preferably single-token answers.

Run a target prompt, transplant the complete activation at one chosen layer-position site from a donor prompt, and then let all unpatched downstream positions recompute normally. This differs from swapping a completed cache and rerunning only the answer token.

Predicted variable-specific effects include:

- transplanting an early $A_1$ representation into $P_{00}$ should favor the hybrid answer $S_{10}=A_1+X_0$;
- transplanting an early $X_1$ representation into $P_{00}$ should favor $S_{01}=A_0+X_1$;
- transplanting a late computed-sum representation from $P_{11}$ should favor $S_{11}$.

The strongest result would be a temporal double dissociation: early sites transmit individual operands and produce recomputed hybrid answers, whereas later sites transmit the completed sum.

Primary outcome measures should include candidate-set log-odds and probability changes, not only top-1 accuracy. For example, an $A$-patch score can compare the predicted hybrid answer against the unchanged target and full donor answers:

\[
\Delta_A
=
\log p(S_{10})
-
\log\!\left(p(S_{00})+p(S_{11})\right).
\]

The exact metric should be fixed before evaluating held-out examples.

## Distinguishing serial from parallel computation

### 1. Path-controlled propagation

Inject a counterfactual operand into an early filler position and test whether it propagates into later filler states and the answer. Then block the answer position’s direct access to the injected site. If later fillers carry the counterfactual forward and still cause the predicted hybrid answer, that supports a serial relay rather than direct answer-side copying.

The complementary intervention blocks communication from the injected filler to later fillers while preserving their access to the question. Recovery under this intervention supports independent rereading or parallel computation.

### 2. Question-cut versus filler-cut

For later filler positions, compare two matched attention interventions:

- **Question-cut:** later fillers cannot attend directly to question tokens but can attend to earlier fillers.
- **Filler-cut:** later fillers can attend to the question but cannot attend to earlier fillers.

Qualitative predictions:

| Algorithm | Question-cut | Filler-cut |
|---|---|---|
| Serial relay | relatively mild damage | severe damage |
| Parallel rereading | severe damage | relatively mild damage |
| Hybrid | partial damage under both | partial damage under both |

For DeepSeek V4 Flash, the intervention must cover every relevant information route, including sliding/local attention and any compressed or latent attention path. Otherwise an apparent negative result may simply reflect an unblocked bypass.

### 3. Conflicting donors

Inject an $A_1$-carrying state at an early filler and an incompatible $A_2$-carrying state later.

- A serial overwrite mechanism predicts a sharp transition toward the later state.
- Independent workers plus aggregation predict mixtures, graded competition, or position-dependent voting.
- Compute-and-broadcast predicts strong dependence on whichever site normally originates the broadcast.

### 4. Topology at fixed token count

Keep the number of filler tokens and total compute approximately fixed while altering allowed communication:

- a connected filler chain;
- isolated fillers that can see the question but not one another;
- grouped filler blocks with communication only inside each block.

A serial mechanism should depend strongly on chain depth. A parallel mechanism should depend more on the number of independent workers and less on chain connectivity.

### 5. Boundary-state sufficiency

Replace a prefix of filler computation with only its final boundary state. If downstream performance is preserved, the prefix may have compressed its computation into a serial sufficient statistic. If performance requires access to several earlier filler states, a distributed or parallel workspace is more plausible.

## Controls

Every causal result should be compared with:

- identity patches;
- same-$X$, different-$A$ donors;
- same-$A$, different-$X$ donors;
- donors differing in both variables;
- unrelated donors;
- random or norm-matched activation controls;
- same-sum, different-decomposition prompts;
- held-out numerical ranges, facts, prompt templates, and paraphrases;
- matched interventions on non-filler positions;
- clean and corrupted baselines to calibrate effect recovery.

Same-sum, different-decomposition examples are particularly useful: they distinguish a representation of the result from representations of the operands or computational route.

## Suggested execution order

1. Reproduce the filler-token performance uplift and establish stable prompts, tokenization, and decoding settings.
2. Cache all four native hidden-state streams and generate logit-lens heatmaps for $A$, $X$, and $A+X$.
3. Use a discovery split for a coarse layer-by-position activation-patching map.
4. Preregister candidate sites and run factorial single-site interchange on held-out examples.
5. Test the early-operand versus late-sum temporal double dissociation.
6. Trace counterfactual propagation while blocking direct answer-side access.
7. Run question-cut/filler-cut and fixed-token-count topology experiments.
8. Use J-Lens or trained probes only where needed to test whether ordinary logit lens is missing latent information.
9. Extend to a two-fact task to test genuine decomposition into parallel subproblems.

## Success criteria

The project would make a strong mechanistic contribution if it shows all of the following:

- a robust behavioral benefit from filler tokens;
- a localized state whose causal interchange favors the preregistered hybrid answer rather than merely the donor answer;
- recomputation of downstream filler states after the transplant;
- a temporal transition from operand-like to result-like causal content;
- a path-controlled double dissociation that discriminates serial communication from independent question rereading.

A weaker but still informative result would be that filler tokens improve behavior without any clean, localized variable representation. That would motivate distributed causal scrubbing, sparse feature analysis, or J-Lens rather than invalidate the project.

## Main risks and pivots

- **No reliable filler uplift:** first audit prompting, tokenization, answer formatting, and model checkpoint; do not proceed to mechanistic interpretation without the behavioral effect.
- **Logit lens is negative but patching is causal:** treat this as evidence of an unaligned latent representation and try J-Lens or a carefully validated probe.
- **Patches only copy the full donor answer:** move earlier in layer/position, use same-sum controls, and require hybrid-answer effects.
- **Attention cuts are bypassed:** enumerate all model-specific information paths and validate each cut with synthetic information-flow controls.
- **One-fact addition cannot distinguish meaningful parallelism:** extend to two facts whose intermediate values can be independently manipulated.

## References

- [Reading Between the Dots](https://arxiv.org/html/2607.03502v1)
- [Public filler-token reasoning experiments](https://github.com/kaleybrauer/filler-token-reasoning)
- [DeepSeek V4 model documentation](https://huggingface.co/docs/transformers/en/model_doc/deepseek_v4)
- [J-Lens / Global Workspace](https://transformer-circuits.pub/2026/workspace/index.html)
- [DeepSeek V4 lens comparison](https://xiangchensong.github.io/blog/2026/jacobian-lens-global-workspace/)
- [Available DeepSeek V4 workspace lenses](https://huggingface.co/camilablank/workspace-lenses/tree/main)
