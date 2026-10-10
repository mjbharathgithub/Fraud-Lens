# How AI-assisted tools were used

FraudLens was built with **Claude Code** (Anthropic's coding agent, used inside VS Code). This note records what the tool did, how its output was checked, and what stayed with the team.

## What the AI tool did

| Stage | Use |
|---|---|
| Requirements | Read the challenge instructions and the project pack, and listed every submission requirement and scoring item |
| Fact-finding | Profiled the raw data before any design (class balance, duplicates, per-day fraud counts, amount patterns) so the plan rested on measured facts |
| Planning | Proposed the evaluation design, the architecture and a build sequence; the team reviewed and approved the plan before any code was written |
| Implementation | Wrote the Python package, the Streamlit app, the scoring API, the tests and the EDA notebook |
| Verification | Ran the training pipeline, the test suite and a scripted walkthrough of the app; rendered every chart and screen and inspected them |
| Documentation | Drafted the README, the solution document, the pitch deck and the architecture diagram from the measured results |

## How the output was checked

AI-written code was not trusted because it ran. Each of these caught a real defect during the build:

- **Reading results critically.** The first logistic-regression baseline had hundreds of transactions tied at a score of exactly 1.0, which made its top 200 arbitrary. It is now ranked on log-odds.
- **Looking at every chart.** The first precision-recall plot was a straight line because the curve had been thinned by index, which threw away the whole high-precision region. The thinning now keeps one point per fraud.
- **Questioning convenient numbers.** Risk-band thresholds derived from training data put all 200 alerts in the top band. They were replaced by explicit policy thresholds whose hit rate is measured and reported.
- **Automated tests (20).** Leakage guards (scaler fitted on training rows only, no resampling at prediction time), the alert-queue contract, the agent's guardrails, and a check that the committed model reproduces the committed scores.
- **A scripted run of the app.** It found a duplicate-element error that only appears when two tabs show the same alert.
- **Honest reporting.** Where the model selected in advance was narrowly beaten on the test day, the document says so rather than switching models.

## AI inside the product

The product itself uses generative AI in one place: an optional local LLM (phi3.5 through Ollama) that rewrites an alert's evidence as a short review note. It is deliberately fenced:

- it sees only a structured evidence object, never raw data;
- the prompt is a versioned file ([prompts/review_note_v1.md](../prompts/review_note_v1.md)) with explicit rules and one worked example;
- code rejects any draft containing a number that is not in the evidence, and falls back to a deterministic template;
- nothing it writes is acted on without an analyst's approval.

## What stayed with the team

- Choosing the use case and approving the plan.
- Deciding what is acceptable to claim: the limits section of the README is part of the submission, not an afterthought.
- Understanding the code well enough to explain every module. [JUDGE_QA.md](JUDGE_QA.md) is the team's preparation sheet for that.
- Presenting the work and answering for it.
