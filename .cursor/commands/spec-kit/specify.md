---
description: Create or update a single umbrella feature specification from a natural language description, optionally referencing multiple existing spec files via @paths.
---

The user input to you can be provided directly by the agent or as a command argument - you **MUST** consider it before proceeding with the prompt (if not empty).

User input:

Input format:
- One natural language feature description
- Zero or more reference file tokens: @relative/or/absolute/path.md
  - All @paths are READ-ONLY references; they DO NOT change the output target
  - Always resolve to absolute paths from the repo root

User input:

$ARGUMENTS

The text the user typed after `/specify` in the triggering message **is** the feature description. Assume you always have it available in this conversation even if `$ARGUMENTS` appears literally below. Do not ask the user to repeat it unless they provided an empty command.

Given that feature description, do this:

1. Parse input
   - Collect all tokens starting with "@" → REF_PATHS[]
   - Resolve each to an absolute path
   - FEATURE_DESC = input text with all @tokens removed and trimmed
   - ERROR if FEATURE_DESC is empty

2. Create feature scaffold (RUN ONCE)
   - Run: `.specify/scripts/bash/create-new-feature.sh --json --no-git "${FEATURE_DESC}"`
   - Parse JSON → BRANCH_NAME, SPEC_FILE (absolute)
   - IMPORTANT: Never run this script more than once per invocation

3. Load template
   - Read `.specify/templates/spec-template.md`

4. Load references (READ-ONLY)
   - For each path in REF_PATHS:
     - Read file content (no writes)
     - **If the file contains markdown links to related specifications:**
       * Note these links in your context
       * Based on FEATURE_DESC, intelligently load relevant linked files using the appropriate tool:

         **Tool Selection Guide:**
         - Use `read_file` when:
           * You have exact file paths from markdown links (e.g., `./endpoints/list.md`)
           * The linked file path is clear and deterministic
           * You need to read a specific known file

         - Use `codebase_search` when:
           * You need to find files by semantic meaning or concept
           * FEATURE_DESC mentions topics without explicit file references
           * You need to discover related files not directly linked
           * Example: FEATURE_DESC mentions "authentication" but no direct link exists

         **Example Decision Tree:**
         1. spec.md links `./endpoints/list.md` + FEATURE_DESC mentions "endpoint list"
            → Resolve relative path to absolute, use `read_file`
         2. spec.md links `./models/endpoint.md` + FEATURE_DESC mentions "endpoint model changes"
            → Use `read_file` with absolute path
         3. FEATURE_DESC mentions "error handling" but no direct links in spec.md
            → Use `codebase_search` with query like "error handling patterns in {service}"
         4. spec.md has many links, FEATURE_DESC is about "authentication flow"
            → Use `codebase_search` or selectively `read_file` only authentication-related linked files

       * Prioritize files whose path or description matches the feature requirements
       * Load only relevant files to avoid context bloat
     - Optionally extract top-level headings to inform Requirements/User Scenarios
   - Prepare a "Referenced Artifacts" section listing all REF_PATHS as bullet items
   - **For modularized specs**: If the main spec.md references detail files (models/, endpoints/, common/), mention these in the "Referenced Artifacts" section for user awareness

5. Write umbrella spec to SPEC_FILE
   - Use template section order exactly
   - Fill placeholders with:
     - Feature Branch, Created, Status, Input = FEATURE_DESC
     - Add a new "Referenced Artifacts" section below the header:
       - List absolute paths (or repo-root relative) of all REF_PATHS
       - Optionally add a brief 1–2 line summary per reference
   - Generate Requirements and User Scenarios from FEATURE_DESC (do not copy tech details from references)
   - Mark ambiguities with [NEEDS CLARIFICATION: …]
   - Do NOT modify any reference files

6. Report completion
   - Print: BRANCH_NAME, SPEC_FILE (absolute), REF_PATHS count, readiness for /plan and /tasks

Notes:
- The script initializes the spec file before writing; branch creation/checkout is disabled.
- All file paths must be absolute and deterministic across runs with same input.

- Feature directory is prefixed with current Git branch when available (e.g., specs/issue-767/<feature>/spec.md).

Examples:

- One umbrella spec referencing serverless + web:
  /spec-kit/specify @specs/serverless/spec.md @specs/web/spec.md "서버리스와 웹 UI 모두 변경되는 피처 설명..."

- Only natural language (no references):
  /spec-kit/specify "사용자 역할별 엔드포인트 가시성 제어 스펙 작성"
