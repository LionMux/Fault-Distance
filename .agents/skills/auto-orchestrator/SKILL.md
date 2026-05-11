---
name: auto-orchestrator
description: Orchestrate complex multi-step research and development workflows automatically. Use when the user asks to build a project from scratch, conduct a literature review with implementation, develop an end-to-end feature across multiple files, or automate any multi-phase task that requires research, planning, coding, and validation in sequence. Also triggers for requests like "create an app", "implement a pipeline", "do a deep research on X and build a prototype", or "refactor this codebase".
metadata:
  author: ProjectTeam
  version: 2.1.0
  category: workflow-automation
---

# Auto-Orchestrator

Use this skill for complex, multi-step requests that blend research, planning, implementation, and validation. This skill provides a structured methodology for coordinating built-in tools, MCP integrations, and domain-specific skills over long-horizon tasks.

## When to use

Activate this skill when a user request matches any of the following patterns:

- **Greenfield development**: "Build a new project from scratch", "Create a web app", "Implement a CLI tool"
- **Research + implementation**: "Do a literature review on X and code a prototype", "Research the best approach for Y and implement it"
- **End-to-end features**: "Add authentication to the API", "Build a caching layer", "Refactor the database layer"
- **Multi-phase automation**: Tasks that require sequential steps across research, design, coding, and testing

Do NOT use for single-step queries like "Explain what a decorator is" or "Fix this typo" — handle those directly.

## Core workflow

Follow these phases in order. Do not skip validation.

### Phase 1: Research

**Goal**: Gather context before committing to a plan.

1. **Assess novelty**: If the topic is new, rapidly evolving, or requires current data, use `exa`/`tavily` web search or activate the `web-search` skill for structured research and source evaluation.
2. **Assess codebase**: If working with an existing project, explore the codebase with `Agent(subagent_type="explore")` or `Glob`/`Grep` to understand structure, conventions, and relevant files.
3. **Synthesize findings**: Summarize key constraints, existing patterns, and external knowledge needed. Save this in your working memory.

**Validation gate**: Before planning, you should be able to answer:
- What technology stack is being used?
- What are the existing conventions (linting, testing, architecture)?
- Are there any hard constraints (dependencies, platform, performance)?

### Phase 2: Planning

**Goal**: Produce an actionable plan with user approval for non-trivial work.

1. **Scope the task**: Break the request into milestones with clear deliverables.
2. **Check for taskmaster**: If `taskmaster` MCP is available, initialize a task plan with `parse_prd` or `set_task_status` and track progress explicitly.
3. **Enter plan mode**: For tasks involving more than 2–3 files, architectural decisions, or unclear requirements, use `EnterPlanMode`. Write the plan to the plan file and present options via `ExitPlanMode`.
4. **Get approval**: Do not start heavy implementation until the user approves the plan (or the task is trivial enough to skip planning).

**Validation gate**: The plan must include:
- Milestones in logical order
- Dependencies between milestones
- Tools/skills to use at each step
- Success criteria for the final output

### Phase 3: Implementation

**Goal**: Execute the plan with high quality and minimal intrusion.

1. **Follow the plan**: Work through milestones sequentially. Update `taskmaster` status as you complete steps.
2. **Write code correctly**:
   - Use `filesystem` tools (`WriteFile`, `StrReplaceFile`, `edit_file`) for all file changes.
   - Follow the `code-style` skill for formatting, naming conventions, and docstring rules.
   - Make minimal changes; do not refactor unrelated code.
3. **Prototype when useful**: For data processing, algorithms, or ML tasks, use `jupyter` to prototype and validate logic before embedding it into the main codebase.
4. **Leverage domain skills**: When the task falls into a specialized domain, explicitly apply the relevant skill:
   - **Web development** → use `web-dev`
   - **ML research / prototyping** → use `ml-research`
   - **ML production systems / MLOps** → use `ml-engineer`
   - **Fault-distance diploma project / COMTRADE** → use `fault-distance`
   - **Structured web research** → use `web-search`
   - **Prompt engineering / LLM optimization** → use `prompt-engineer`

**Validation gate**: After each milestone, verify that:
- Files were created/modified as intended
- The code follows project conventions (including `code-style` rules)
- No syntax errors were introduced

### Phase 4: Validation

**Goal**: Prove the implementation works.

1. **Run the code**: Execute scripts, run tests, or start services using `Shell`.
2. **Iterate on failures**: If tests fail or runtime errors occur, read the logs, diagnose the root cause, fix the code, and re-run.
3. **Check outputs**: For data pipelines, inspect outputs. For web apps, verify routes or take screenshots with `playwright` if needed.
4. **Ensure test passage**: If the project has a test suite, run it and do not finish until it passes (or document why a failure is expected).

**Validation gate**: You must have objective evidence that the implementation works — a passing test, a successful run, or a verified output.

### Phase 5: Error handling & iteration

**Goal**: Recover gracefully and keep moving forward.

1. **Log issues briefly**: If a tool call fails, summarize the error in one sentence.
2. **Retry with correction**: Fix the underlying issue (wrong path, missing dependency, incorrect parameter) and retry the call.
3. **Escalate if blocked**: If you are stuck for more than 2–3 attempts, explain the blocker to the user and ask for direction.
4. **Never stop completely**: Unless explicitly told to stop, always proceed with the next actionable step.

## Examples

### Example 1: Building a new project from scratch

**User says**: "Build a Python CLI tool that fetches weather data and prints a summary."

**Actions**:
1. Research current weather APIs using `web-search`.
2. Plan the CLI structure (argparse, API client, formatter, tests).
3. Create files: `weather_cli.py`, `requirements.txt`, `README.md`.
4. Follow `code-style` for Python formatting.
5. Validate by running `python weather_cli.py --city Berlin`.

### Example 2: Literature review with implementation

**User says**: "Research the best image segmentation models for medical data and implement a small demo."

**Actions**:
1. Search for recent papers and benchmarks with `web-search` or `exa`.
2. Summarize top 2–3 approaches.
3. Plan a minimal demo using the best-performing accessible model.
4. Implement the demo in a Jupyter notebook or Python script, following `ml-research` and `code-style`.
5. Validate by running inference on a sample image.

### Example 3: End-to-end feature development

**User says**: "Add JWT authentication to this FastAPI project."

**Actions**:
1. Explore the codebase to find existing auth patterns and main router files.
2. Plan the changes (dependencies, middleware, protected routes, tests).
3. Implement: add `pyjwt` to requirements, create auth module, protect routes.
4. Validate by running the test suite or manual API requests.

## Troubleshooting

### Issue: Skill triggers on simple queries

**Cause**: The user said something that could be interpreted as complex but is actually simple (e.g., "How do I build a web app?" as a conceptual question).

**Fix**: Ask one clarifying question. If the user just wants an explanation, answer directly without entering the full orchestration workflow.

### Issue: MCP tool calls fail repeatedly

**Cause**: Server disconnected, invalid parameters, or missing authentication.

**Fix**:
1. Verify the MCP server is available.
2. Check that parameter names and types match the MCP schema.
3. If authentication is required, prompt the user for the missing credential.

### Issue: Plan mode rejected by user

**Cause**: User wants faster iteration or already knows what they want.

**Fix**: Switch to lightweight planning — briefly state your next 2–3 steps in chat and proceed immediately. Do not force formal plan mode on small changes.

### Issue: Implementation drifts from the plan

**Cause**: Edge cases discovered during coding.

**Fix**: If the drift is small, adapt and note it. If the drift is large (new architecture, extra dependencies), pause and confirm the new direction with the user.

## Performance notes

- **Quality over speed**: Take the time to do research and validation thoroughly.
- **Do not skip validation steps**: Every non-trivial change should be run or tested.
- **Use subagents for exploration**: Delegate read-only codebase exploration to `Agent(subagent_type="explore")` to save context window.
- **Track progress explicitly**: When `taskmaster` is available, update task status after each milestone so the user can follow along.
- **Delegate to domain skills**: Do not recreate domain expertise inside this skill; invoke `web-dev`, `ml-research`, `ml-engineer`, `fault-distance`, `web-search`, or `code-style` as needed.
