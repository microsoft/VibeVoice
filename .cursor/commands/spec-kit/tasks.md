---
description: Generate an actionable, dependency-ordered tasks.md for the feature based on available design artifacts.
---

The user input to you can be provided directly by the agent or as a command argument - you **MUST** consider it before proceeding with the prompt (if not empty).

User input:

$ARGUMENTS

1. Run `.specify/scripts/bash/check-prerequisites.sh --json --no-git` from repo root and parse FEATURE_DIR and AVAILABLE_DOCS list. All paths must be absolute.
2. Load and analyze available design documents:
   - Always read plan.md for tech stack and libraries
   - **Always parse plan.md's "Affected Spec Areas" section** to get spec update details
   - IF EXISTS: Read data-model.md for entities
   - IF EXISTS: Read contracts/ for API endpoints
   - IF EXISTS: Read research.md for technical decisions
   - IF EXISTS: Read quickstart.md for test scenarios

   Note: Not all projects have all documents. For example:
   - CLI tools might not have contracts/
   - Simple libraries might not need data-model.md
   - Generate tasks based on what's available

3. Generate tasks following the template:
   - Use `.specify/templates/tasks-template.md` as the base
   - Replace example tasks with actual tasks based on:
     * **Setup tasks**: Project init, dependencies, linting
     * **Test tasks [P]**: One per contract, one per integration scenario
     * **Core tasks**: One per entity, service, CLI command, endpoint
     * **Integration tasks**: DB connections, middleware, logging
     * **Polish tasks [P]**: Unit tests, performance, docs
     * **Spec Update tasks**: Update specs/{service}/spec.md for affected services

4. Task generation rules:
   - Each contract file → contract test task marked [P]
   - Each entity in data-model → model creation task marked [P]
   - Each endpoint → implementation task (not parallel if shared files)
   - Each user story → integration test marked [P]
   - Different files = can be parallel [P]
   - Same file = sequential (no [P])

5. Order tasks by dependencies:
   - Setup before everything
   - Tests before implementation (TDD)
   - Models before services
   - Services before endpoints
   - Core before integration
   - Everything before polish
   - Spec updates after all implementation and testing

6. Include parallel execution examples:
   - Group [P] tasks that can run together
   - Show actual Task agent commands

7. Create FEATURE_DIR/tasks.md with:
   - Correct feature name from implementation plan
   - Numbered tasks (T001, T002, etc.)
   - Clear file paths for each task
   - Dependency notes
   - Parallel execution guidance

   **Spec Documentation Updates**:
   - **PRIMARY SOURCE**: Parse plan.md's "Affected Spec Areas" section (YAML format)
     * Extract service name, structure type, and file list
     * For each file entry: path, reason, and changes are already specified
     * Generate one task per file with exact path and change description
     * If "Affected Spec Areas" section is missing or empty, fall back to directory detection (legacy behavior)

   - **FALLBACK**: If "Affected Spec Areas" not found in plan.md (legacy support):
     ⚠️ **Important**: This FALLBACK only supports backend services with standard structure.
     For non-backend services (web, etc.) or special structures, "Affected Spec Areas" is required.

     * Analyze plan.md to identify affected services
     * For each backend service, determine which spec files need updates:

       **Structure detection (backend services only)**:
       - Check if specs/{service}/ contains all 3 subdirectories:
         * models/ (must exist)
         * endpoints/ (must exist)
         * common/ (must exist)
       - If all 3 exist → modularized service
       - Otherwise → single-file service

       **For modularized backend services**:
       - Identify specific files to update based on change type
       - Update main spec.md only if architecture/overview changes
       - Update specific detail files for focused changes

       **For single-file services**:
       - Update specs/{service}/spec.md directly

   - **Change type → File mapping** (used only in FALLBACK mode for backend services):
     * New/modified API endpoint → specs/{service}/endpoints/{endpoint-name}.md
     * Database schema change → specs/{service}/models/{model-name}.md
     * Error handling change → specs/{service}/common/error-handling.md
     * Auth/permission change → specs/{service}/common/authentication.md
     * CORS/config change → specs/{service}/common/cors-policy.md
     * Architecture change → specs/{service}/spec.md (main index)
     * New workload type → specs/{service}/spec.md + new model file if needed

     ⚠️ **Note**: This mapping assumes backend service structure.
     Frontend services (web) or special cases require "Affected Spec Areas" in plan.md.

   - **Task structure for modularized specs**:
     ```
     T0XX Update {Service} {Specific Area} Specification [P]
     - Files:
       * specs/{service}/endpoints/{name}.md (if API changed)
       * specs/{service}/models/{name}.md (if model changed)
       * specs/{service}/common/{aspect}.md (if common concern changed)
       * specs/{service}/spec.md (if architecture/overview changed)
     - Changes: [Specific updates from plan - be concrete]
     - Examples:
       * "Add status filter parameter to Query Parameters in endpoints/list.md"
       * "Add metrics_enabled field to Task model in models/task.md"
       * "Update main spec.md Last Updated date and version"
       * "Add rate limiting section to common/authentication.md"
     - Dependency: After implementation tasks (T0XX, T0YY)
     ```

   - **Task structure for single-file specs**:
     ```
     T0XX Update {Service} Specification [P]
     - File: specs/{service}/spec.md
     - Section: {Architecture|Data Model|API Surface|Operational Concerns|etc.}
     - Updates: [Specific changes from plan - be concrete]
     - Example: "Add new endpoint GET /api/tasks/{id}/metrics to API Surface"
     - Dependency: After implementation tasks (T0XX, T0YY)
     ```

   - **Task generation logic**:
     * **PREFERRED**: Use "Affected Spec Areas" from plan.md (provides exact paths and change descriptions)
     * **FALLBACK**: If "Affected Spec Areas" missing, detect structure with list_dir/glob_file_search
     * Generate one task per file listed in "Affected Spec Areas"
     * Include the "reason" and "changes" from plan.md in task description
     * Benefits: eliminates duplicate analysis, ensures consistency between plan and tasks

   - Mark spec update tasks as [P] since different files/services are independent
   - Place spec tasks after all implementation and testing (ensure docs match reality)
   - Check for cross-service impacts:
     * Backend API changes → Update both backend detail files AND web/spec.md API Surface
     * New features → Update affected feature in web/spec.md Feature Coverage
     * Shared models → Update all services referencing that model

   - **Example of granular spec updates for serverless**:
     ```
     T050 Update Serverless Endpoint List API Spec [P]
     - File: specs/serverless/endpoints/list.md
     - Changes:
       * Add status filter parameter to Query Parameters section
       * Update Request Example with new filter usage
       * Add validation rules for status enum values
     - Dependency: After T040 (Implement status filter)

     T051 Update Serverless Endpoint Model Spec [P]
     - File: specs/serverless/models/endpoint.md
     - Changes:
       * Add status field to Fields table with type and description
       * Document valid status enum values (pending, running, stopped)
       * Add status to Constraints section
     - Dependency: After T038 (Add status column migration)

     T052 Update Serverless Main Spec [P]
     - File: specs/serverless/spec.md
     - Changes:
       * Update "Last Updated" date to current date
       * Update version to 2.1
       * Note new filtering capability in Quick Reference
     - Dependency: After T050, T051
     ```

Context for task generation: $ARGUMENTS

The tasks.md should be immediately executable - each task must be specific enough that an LLM can complete it without additional context.
