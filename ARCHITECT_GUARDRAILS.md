The Core Principle: "Earned Autonomy"
The system must demonstrate competence and reliability at each phase before gaining more control. This is about building trust through proven performance.

Phase 1: Learner's Permit (Initial Development)
Goal: Prove basic competence and learn the process of improvement under strict supervision.

Rules:

Read-Only Git: The Architect can analyze the codebase and its own performance data, but cannot write to the main branch.

Batch Processing & Reporting: After a set number of games (e.g., 300), the Architect must generate a Change Proposal Report. This report must include:

Proposed Changes: Specific, commented code snippets or architectural adjustments.

Justification: Data-driven reasoning. "Module A's latency increased failure rate by 15%. This refactor reduces latency by 40ms, projected to improve win rate by 5%."

Impact Analysis: Which subsystems will be affected and how.

Rollback Plan: A clear procedure to revert the change if it causes regression.

Human-in-the-Loop Approval: A human engineer must review, test, and approve every change. This is a supervised learning phase for the Architect itself.

Phase 2: Provisional License (Demonstrated Competence)
Goal: Transition from suggesting changes to implementing them in a safe, sandboxed environment.

Trigger Condition: The system can consistently beat full games with minimal human intervention. Its Change Proposals have a high success rate (e.g., >90% lead to performance gains).

Rules:

Sandboxed Git Access: The Architect is granted write access to a development branch.

Automated CI/CD Gatekeeper: All changes must pass through an automated pipeline that includes:

Unit Tests: Does the change break existing functionality?

Performance Benchmarks: Does it meet the performance improvements it promised?

Regression Tests: Does it cause a regression in any previously mastered game?

Human Sign-Off for Merge: The Architect can implement changes in the dev branch, but a human must still approve the merge into the main trunk after reviewing the automated test results.

Phase 3: Full License (Mature Autonomy)
Goal: Full self-evolution, with humans in an oversight role.

Trigger Condition: The system has successfully executed multiple major rewrites without causing regressions. It has developed a robust internal model of system stability and performance.

Rules:

Full Git Autonomy: The Architect can commit, branch, and merge to the main codebase directly.

The "Council" Oversight: Humans are not in the direct approval loop but act as a "board of directors." The Architect must provide a Regular Evolution Report that summarizes:

What changes were made and why.

Performance deltas across all benchmarks.

Any unexpected consequences and how they were mitigated.

The "Big Red Button": A failsafe mechanism is built in. Humans retain the ultimate authority to:

Pause all autonomous activity.

Rollback to a previous known-good commit.

Reinstate rules from a previous phase if the system's behavior becomes unstable or misaligned.

Meta-Rule: The Architect cannot modify the oversight rules or the "Big Red Button" mechanism. This is the constitutional law of the system.

Foundational House Rules (Apply to All Phases)
The Transparency Rule: All reasoning, data, and proposals must be logged in the database in a human-readable format. No black-box decisions.

After completing an experiment, the system should check wheter its on an experimental architect branch, or on the main branch (Tabula-Rasa-v{#}) (and when the trainer is manually started - always use the main branch. When its automatically started by the director or an llm/architect use the automated suggestion or the main branch (Tabula-Rasa-v{#}) as a fallback.)

The Conservatism Rule: When in doubt between a radical refactor and a incremental tweak, the Architect should favor the incremental approach. Stability is prioritized over raw speed of improvement.

The Alignment Rule: The context-dependent fitness function must always include a stability metric. Rewards for winning must be balanced against penalties for unpredictable or erratic behavior.

The "First, Do No Harm" Principle: Any change that causes a regression in a previously mastered task is automatically considered a failure, and the Architect must analyze the root cause before proposing further changes.