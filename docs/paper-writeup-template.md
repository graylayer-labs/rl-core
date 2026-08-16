# Paper reproduction template

**Tone:** Write as technical documentation, not a personal narrative. Remove "I", "my", "we". Assume reader understands RL basics. No marketing language or hype.

## Why this paper

Why reproduce this work? What makes it foundational or necessary to understand? What algorithms build on it?

Keep it concise. No personal motivation needed—focus on the paper’s significance.

## What the paper does

Explain the problem, central idea, and key mechanisms in engineering terms. Use one diagram, equation, or pseudocode only if it materially clarifies the idea.

## Claim under test

State the paper’s core claim in one sentence. Example: "Algorithm X learns task Y from input Z and achieves performance > baseline."

## Implementation

Reference the code location (`rl_core.integrations.AlgorithmName`). Describe key design choices and any deliberate deviations from the paper.

## Reproduction experiment

Link to `reproduction.yaml` for the frozen protocol. Summarize: testbed, seeds, compute budget, success criterion.

## Results

Lead with the result. Show learning curves, aggregate scores, uncertainty, and runtime. Clearly label each type of evidence: smoke test, pilot, partial run, or qualifying.

If the experiment hasn’t run, say so. Don’t fill this with expected results.

## Key insights

What did the implementation reveal about the algorithm? What’s non-obvious?

- What failed and what did that expose?
- Which implementation detail mattered more than expected?
- What assumptions from the paper held up?
- What remains unresolved?

## Next steps

What experiment or paper comes after this? State which assumption or unanswered question drives it.
