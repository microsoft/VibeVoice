
# Implementation Plan: [FEATURE]

**Branch**: `[###-feature-name]` | **Date**: [DATE] | **Spec**: [link]
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from file system structure or context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
[Extract from feature spec: primary requirement + technical approach from research]

## Technical Context
**Language/Version**: [e.g., Python 3.11, Swift 5.9, Rust 1.75 or NEEDS CLARIFICATION]
**Primary Dependencies**: [e.g., FastAPI, UIKit, LLVM or NEEDS CLARIFICATION]
**Storage**: [if applicable, e.g., PostgreSQL, CoreData, files or N/A]
**Testing**: [e.g., pytest, XCTest, cargo test or NEEDS CLARIFICATION]
**Target Platform**: [e.g., Linux server, iOS 15+, WASM or NEEDS CLARIFICATION]
**Project Type**: [single/web/mobile - determines source structure]
**Performance Goals**: [domain-specific, e.g., 1000 req/s, 10k lines/sec, 60 fps or NEEDS CLARIFICATION]
**Constraints**: [domain-specific, e.g., <200ms p95, <100MB memory, offline-capable or NEEDS CLARIFICATION]
**Scale/Scope**: [domain-specific, e.g., 10k users, 1M LOC, 50 screens or NEEDS CLARIFICATION]

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Data Structure First (Principle I)
- [ ] If modifying existing code: Database schema analyzed before code changes?
- [ ] Entity relationships and constraints documented?
- [ ] Performance implications assessed (indexes, N+1 queries)?
- [ ] N/A - This is a new feature

### API Contracts (Principle IV)
- [ ] OpenAPI specs defined in `contracts/` before implementation?
- [ ] Request/response schemas validated at boundaries (Pydantic/Zod)?
- [ ] Error envelope follows standard format `{ data?, error?, meta? }`?
- [ ] Contract tests exist for all public endpoints?

### Test-Driven Change Management (Principle VII)
- [ ] Tests written BEFORE implementation?
- [ ] Test pyramid followed (unit > integration > e2e)?
- [ ] All tests currently FAILING (no premature implementation)?

### Security by Default (Principle VI)
- [ ] Authentication/authorization requirements identified?
- [ ] Input validation beyond schema (length, regex, sanitization)?
- [ ] Secrets management strategy defined (no hardcoded secrets)?
- [ ] Rate limiting requirements documented?

### Multi-Language Standards (Principle III)
- [ ] **Python**: Alembic migration created? FastAPI `response_model` used?
- [ ] **TypeScript**: No `any` types? React hooks pattern? Zod validation?
- [ ] **Go**: Standard layout? Explicit error returns?

### Kubernetes Native (Principle V - if applicable)
- [ ] Pod `resources.requests/limits` defined?
- [ ] Liveness/readiness probes specified?
- [ ] GPU/NPU nodeSelector/tolerations configured?
- [ ] N/A - No Kubernetes resources

### Monorepo Consistency (Principle II - if applicable)
- [ ] Turbo pipeline respects dependency graph?
- [ ] Changes limited to single workspace or properly coordinated?

---

**Violations & Justifications** (fill only if any checks failed):

| Principle | Why Violation Accepted | Remediation Timeline |
|-----------|------------------------|---------------------|
| [e.g., TDD] | [e.g., exploratory spike] | [e.g., tests added by T010] |

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., ai-platform/frontend). The delivered plan must
  not include Option labels.
-->
```
# [REMOVE IF UNUSED] Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# [REMOVE IF UNUSED] Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# [REMOVE IF UNUSED] Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure: feature modules, UI flows, platform tests]
```

**Structure Decision**: [Document the selected structure and reference the real
directories captured above]

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh cursor`
     **IMPORTANT**: Execute it exactly as specified above. Do not add or remove any arguments.
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Affected Spec Areas
*This section records which spec files need updates after implementation completes.*
*The /tasks command will parse this section to generate precise spec update tasks.*

**Format**:
```yaml
service: [service-name]  # e.g., serverless, pods, storage
structure: [modularized|single-file]  # detected from specs/{service}/ directory
files:
  - path: specs/{service}/{subdir}/{file}.md
    reason: [why this file needs update]
    changes: [brief description of what sections/content to update]
```

**Example for modularized service** (serverless):
```yaml
service: serverless
structure: modularized
files:
  - path: specs/serverless/endpoints/list.md
    reason: "Adding status filter parameter to endpoint"
    changes: "Add status to Query Parameters, update Request Example, add validation rules"
  - path: specs/serverless/models/endpoint.md
    reason: "New status field in Endpoint model"
    changes: "Add status field to Fields table with enum values (pending, running, stopped)"
  - path: specs/serverless/spec.md
    reason: "Version and metadata update"
    changes: "Update Last Updated date, bump version to 2.1, note new filtering capability"
```

**Example for single-file service** (benchmarks):
```yaml
service: benchmarks
structure: single-file
files:
  - path: specs/benchmarks/spec.md
    reason: "New benchmark type support"
    changes: "Add new type to Data Model section, update API Surface with new endpoints"
```

**Instructions for filling this section**:
1. Identify all services affected by this feature (pods, serverless, storage, web, etc.)

2. For each service, automatically detect structure type:

   **Backend services** (auth, pods, serverless, storage, datasets, finetune, etc.):
   - Check for: models/, endpoints/, common/ subdirectories
   - If all 3 exist → `structure: modularized`
   - Otherwise → `structure: single-file`

   **Frontend services** (web):
   - Check for service-specific subdirectories (e.g., ui/, playwright/, language/)
   - If has organized subdirectories → `structure: modularized`
   - If only spec.md → `structure: single-file`

   **General rule**:
   - Use list_dir on specs/{service}/ to check directory structure
   - Any service with meaningful subdirectories (not just spec.md) is modularized
   - Each service type may have different subdirectory patterns

3. For modularized services, map changes to specific files:

   **Backend services:**
   - API changes → endpoints/{name}.md
   - Model changes → models/{name}.md
   - Error handling → common/error-handling.md
   - Auth/permissions → common/authentication.md
   - Architecture → spec.md

   **Frontend services (web):**
   - UI component changes → ui/{component}.md
   - E2E test changes → playwright/{test}.md
   - i18n changes → language/{locale}.md
   - Architecture → spec.md

   **Note**: Each service may have unique subdirectory structure. Adapt file mapping accordingly.

4. List exact file paths, not patterns

5. Provide concrete change descriptions (what sections, what content)

---

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each contract → contract test task [P]
- Each entity → model creation task [P]
- Each user story → integration test task
- Implementation tasks to make tests pass
- **Parse "Affected Spec Areas" above to generate spec update tasks**

**Ordering Strategy**:
- TDD order: Tests before implementation
- Dependency order: Models before services before UI
- Mark [P] for parallel execution (independent files)
- Spec update tasks come last (after implementation verified)

**Estimated Output**: 25-30 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [ ] Phase 0: Research complete (/plan command)
- [ ] Phase 1: Design complete (/plan command)
- [ ] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [ ] Initial Constitution Check: PASS
- [ ] Post-Design Constitution Check: PASS
- [ ] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

---
*Based on Constitution v2.0.0 - See `.specify/memory/constitution.md`*
