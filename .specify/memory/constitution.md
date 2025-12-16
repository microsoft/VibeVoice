<!--
Sync Impact Report
- Version change: 1.0.0 → 2.0.0
- Modified principles: Scope expanded from Serverless Backend only to full-stack AI Platform
- Added sections:
  * Monorepo Management & Workspace Structure
  * Multi-Language Consistency (Python/TypeScript/Go)
  * Frontend Engineering Standards
  * MCP Agent Development
- Removed sections: None (previous content generalized)
- Templates requiring updates:
  * .specify/templates/plan-template.md ✅ updated (Constitution Check expanded + version bump)
  * .specify/templates/spec-template.md ✅ no change (not constitution-aware)
  * .specify/templates/tasks-template.md ✅ no change (already TDD-compliant)
- Follow-up TODOs:
  * Review workspace rules alignment with new principles
  * Update agent-specific templates to reference new constitution sections
-->

# AI Platform Web UI Constitution

## Core Principles

### I. Data Structure First (Code Follows Schema)

**When modifying existing code** (not new features), database schema MUST be analyzed before code changes:
1. Understand table structures and relationships in `app/models/`
2. Trace data usage across API endpoints and business logic
3. Ensure changes maintain referential integrity and performance (indexes, N+1)

**Rationale**: Prevents data corruption, schema drift, and technical debt by aligning code with persistent state.

### II. Monorepo as Single Source of Truth

- **Workspace Structure**: pnpm workspace defines the frontend application. All shared code is consolidated within `ai-platform/frontend/`.
- **Build Orchestration**: Turbo manages build, lint, test, type-check across all workspaces. Task pipelines MUST respect dependency graphs.
- **Version Consistency**: Lock files (`pnpm-lock.yaml`) are canonical. `packageManager` field enforces pnpm@10.12.1 via Corepack.

**Rationale**: Ensures reproducible builds, eliminates version conflicts, and scales to multi-team development.

### III. Multi-Language Consistency

Each language stack maintains independent standards while sharing design patterns:

**Python (FastAPI Backends)**:
- Alembic is the single source of truth for DB schema
- Pydantic models for validation; FastAPI `response_model` required
- Tools: ruff (lint+isort), black (format), mypy (types), bandit (security)
- REST API versioning at `/v1`, OpenAPI docs MUST match implementation

**TypeScript (React Frontend + Node.js Services)**:
- TypeScript strict mode; no `any` types except explicitly justified
- React 18 patterns: hooks over classes, composition over inheritance
- Tools: ESLint, Prettier, TypeScript compiler
- UI: TailwindCSS + 9ui component library; see styling guidelines in workspace rules

**Go (MCP Agents)**:
- Standard Go project layout (`cmd/`, `internal/`, `pkg/`)
- Error handling via explicit returns; context propagation for cancellation
- Tools: golangci-lint, go fmt, go vet
- gRPC/HTTP contracts defined via OpenAPI or protobuf

**Rationale**: Language-specific best practices prevent Lowest Common Denominator syndrome while maintaining architectural coherence.

### IV. API Contracts as Interfaces

- **Contract-First**: OpenAPI specs (YAML/JSON) in `docs/api/` or embedded via FastAPI autodoc. Frontend API clients are generated from specs.
- **Versioning**: Breaking changes MUST increment version (`/v1` → `/v2`). Deprecations follow 2-release policy.
- **Validation**: Request/response schemas validated at boundaries (Pydantic, Zod). Contract tests MUST exist for all public endpoints.
- **Error Envelopes**: Standard format across all services: `{ data?, error?, meta? }`. 4xx for client errors, 5xx for server errors.

**Rationale**: Decouples frontend/backend development, enables parallel work, and prevents runtime integration failures.

### V. Kubernetes Native Operations

**Resource Management**:
- Pod `resources.requests/limits` are MANDATORY
- GPU/NPU workloads MUST use `nodeSelector`/`tolerations` with standardized labels
- Storage: PVC provisioning via NFS Server Provisioner (`tkai-nfs-individual` StorageClass)

**Deployment Standards**:
- Helm charts are the single source of infrastructure truth (`helm/ai-platform/`)
- Liveness/Readiness probes REQUIRED; `terminationGracePeriod` ≥ 30s for stateful services
- Immutable image tags (digest-based); no `:latest` in production
- Secrets via Kubernetes Secrets or external KMS; NEVER in code or values files

**Observability**:
- JSON-structured logs with request IDs (correlation across services)
- Prometheus metrics for throughput, error rates, latency (p50/p95/p99)
- Distributed tracing via OpenTelemetry (when applicable)

**Rationale**: Aligns with cloud-native operational patterns, enables auto-scaling and self-healing, ensures production reliability.

### VI. Security by Default

- **Authentication & Authorization**: All non-public endpoints MUST verify tokens/sessions. RBAC for admin operations.
- **Input Validation**: Beyond schema validation (length limits, regex whitelists, sanitization).
- **Secrets Management**: Rotate regularly; audit access. No secrets in environment variables visible to logs.
- **Rate Limiting**: Per-user/token quotas to prevent abuse.
- **Dependency Scanning**: `pip-audit` (Python), `npm audit` (Node), `govulncheck` (Go) in CI.

**Rationale**: Proactive security prevents breaches and ensures compliance with enterprise security policies.

### VII. Test-Driven Change Management

**Test Pyramid**:
- **Unit**: Fast, isolated, covering business logic (>70% of test suite)
- **Integration**: Database, external APIs, service interactions
- **E2E**: Critical user flows only (Playwright for frontend)

**Change Workflow**:
1. Feature spec → implementation plan → tasks (via `.specify/templates/`)
2. Write failing tests BEFORE implementation (TDD)
3. Implement minimum code to pass tests
4. Refactor with test coverage
5. Update documentation (API docs, quickstart guides)

**Bug Fixes**: MUST include a reproducing test that fails before the fix.

**Rationale**: Tests as living documentation; regression prevention; confidence in refactoring.

## Engineering Guardrails

### Code Quality Automation

**Pre-Commit Hooks** (MANDATORY before commits):
- Python: ruff, black, mypy, bandit
- TypeScript: ESLint, Prettier, TypeScript compiler
- Go: golangci-lint, gofmt

**CI Checks** (REQUIRED for merge):
- All tests pass (unit, integration, contract)
- Linting and formatting pass
- Type checks pass (mypy, tsc, go vet)
- Security scans clean (no HIGH/CRITICAL CVEs)
- Alembic migrations dry-run succeeds

**Branch Protection**:
- Branch flow: `issue/#<NUM>` → `dev` → `main`
- Conventional Commits format (see workspace rules)
- Squash merging for feature branches; linear history on `main`

### API Development Standards

**FastAPI Services**:
- Explicit `response_model` for all handlers
- Dependency injection for DB sessions, auth contexts
- Exception handlers mapped to HTTP status codes centrally
- Server-side timeouts (e.g., 30s for standard requests)
- Idempotency-Key support for mutating operations

**React Frontend**:
- Component structure: feature-based folders (`features/pods/`, `features/serverless/`)
- State management: Zustand for global, React hooks for local
- Form handling: react-hook-form with Zod validation
- Error boundaries for graceful failures
- Accessibility: WCAG AA compliance (contrast ratios, focus management, ARIA labels)

### Data & Migrations

**Alembic (Python Services)**:
- All DDL changes via Alembic migrations; NEVER raw SQL in code
- Migrations MUST be reversible (implement `downgrade()`)
- Prefer NOT NULL + CHECK constraints over application-only validation
- Index high-cardinality columns used in WHERE/JOIN clauses

**Database Best Practices**:
- Avoid N+1 queries: use eager loading (SQLAlchemy `joinedload`)
- Connection pooling tuned for workload (e.g., 10-20 connections per service)
- Read replicas for reporting queries (when scale demands)

### Infrastructure as Code

**Helm Chart Standards**:
- Values hierarchy: `values.yaml` (defaults) → `values-dev.yaml` → `values-prod.yaml`
- Required chart metadata: version, appVersion, maintainers
- Templates MUST include: Deployment, Service, ConfigMap, Secret, Ingress (if applicable)
- Resource quotas and limits defined per environment

**GitOps Deployment**:
- Environments: `dev` (auto-deploy from `dev` branch) → `staging` (manual promotion) → `prod` (approval gate)
- Argo Rollouts for progressive delivery (canary, blue-green)
- Automated rollback on SLO violations (error rate >1%, latency p95 >500ms)

### Release & Deployment

**Merge Gates**:
- All CI checks green (tests, lint, types, security)
- Code review approved (at least 1 maintainer)
- Alembic migration present if schema changed
- API docs updated if endpoints changed
- Changelog entry added (user-facing changes only)

**Deployment Strategy**:
- Docker images: multi-stage builds, rootless, signed SBOMs
- Progressive rollout: 10% → 50% → 100% over 1 hour
- Feature flags for risky changes (instant rollback without redeploy)
- Runbook update if operational procedures change

## Platform & Operations

### Service Architecture

**Backend Services** (Python FastAPI):
- pods (8000), finetune (8001), serverless (8002), datasets (8003)
- resources (8004), leaderboard (8005), my_templates (8006), storage (8007)
- Each service: independent PostgreSQL schema, isolated deployments

**Frontend** (TypeScript React):
- Web UI (8080): Vite dev server → Nginx production
- API clients auto-generated from backend OpenAPI specs

**MCP Agents** (Go Microservices):
- cloud-suite, costs, mytemplates, pod-solver, resources, serverless, workloads
- gRPC or REST endpoints; deployed as separate Deployments

**Databases**:
- PostgreSQL: AI Platform backends (connection pooling via PgBouncer)
- MongoDB: vLLM Benchmark services (replica set for HA)

### Observability & SLOs

**Logging**:
- JSON-structured: `{ timestamp, level, message, request_id, service, ...context }`
- Correlation: request IDs propagated via headers (`X-Request-ID`)
- Sensitive data masked (tokens, passwords, PII)

**Metrics** (Prometheus):
- RED: Request rate, Error rate, Duration (p50/p95/p99)
- Resource: CPU, memory, disk, GPU utilization per pod
- Business: Active users, job success rate, storage usage

**SLOs**:
- API latency: p95 <300ms (read), p95 <500ms (write)
- Availability: 99.5% uptime per service (allows ~3.6h downtime/month)
- Error budget: 0.5% of requests; budget exhaustion blocks non-critical releases

### Security Operations

**Access Control**:
- Least-privilege RBAC: services run as dedicated ServiceAccounts
- No `cluster-admin` for application workloads
- Secrets rotation: 90-day cycle with audit trail

**Compliance**:
- Administrative actions logged (token issue/revoke, scale changes, data access)
- Security scans weekly; HIGH/CRITICAL CVEs patched within 7 days
- Quarterly audit of logging, metrics, and security controls

## Governance

### Authority & Scope

This constitution governs engineering practices for the **AI Platform Web UI** project, encompassing:
- Frontend web application (`ai-platform/frontend/`)
- Backend microservices (`ai-platform-backend/`)
- MCP agents (`agents/`)
- Infrastructure as Code (`helm/`, `infra/`)

### Amendment Procedure

1. Propose changes via PR updating `.specify/memory/constitution.md`
2. Include **Sync Impact Report** (updated template references, affected principles)
3. Require approval from:
   - At least 1 backend maintainer (Python services)
   - At least 1 frontend maintainer (TypeScript)
   - At least 1 platform/infra maintainer (Kubernetes/Helm)
4. Document migration path if amendment affects existing code
5. Update workspace rules (`.cursor/rules/`) if conflicts arise

### Versioning Policy

**Semantic Versioning**:
- **MAJOR**: Backward-incompatible changes (e.g., removing a principle, changing branch strategy)
- **MINOR**: New principles or materially expanded sections
- **PATCH**: Clarifications, typo fixes, non-semantic edits

**Metadata**:
- **Ratified**: Original adoption date (preserved across amendments)
- **Last Amended**: Date of most recent merged change

### Compliance & Review

**PR Review Checklist**:
- [ ] Changes comply with applicable constitution principles
- [ ] Deviations documented with time-bounded exception and rationale
- [ ] Tests added/updated per TDD principle
- [ ] Documentation updated (API docs, quickstart guides)
- [ ] Security and performance implications assessed

**Periodic Audits**:
- **Quarterly**: Review adherence to observability, security, and testing principles
- **Bi-annually**: Evaluate constitution relevance against evolving project needs
- **Findings**: Documented in `docs/governance/audit-reports/`

### Exceptions & Technical Debt

**Exception Process**:
1. Document in PR description: principle violated, business justification, remediation timeline
2. Tag as `tech-debt` in issue tracker
3. Require explicit approval from area maintainer
4. Re-evaluate at next quarterly review

**Debt Repayment**:
- Track in `docs/technical-debt.md` with priority (HIGH/MEDIUM/LOW)
- Allocate 20% of sprint capacity to debt reduction
- Block new features if HIGH priority debt exceeds threshold (5 items)

---

**Version**: 2.0.0 | **Ratified**: 2025-09-29 | **Last Amended**: 2025-10-01
