## Plan

Helps you plan before coding. Creates detailed strategies to make development smoother.

### Usage

```bash
# Request Plan Mode from Claude
"Create an implementation plan for [implementation content]"
```

### Basic Examples

```bash
# Implementation plan for new feature
"Create an implementation plan for user authentication functionality"

# System design plan
"Create an implementation plan for microservice splitting"

# Refactoring plan
"Create a refactoring plan for legacy code"
```

### Collaboration with Claude

```bash
# Complex feature implementation
"Create an implementation plan for chat functionality, including WebSocket, real-time notifications, and history management"

# Database design
"Create a database design plan for an e-commerce site, including product, order, and user management"

# API design
"Create an implementation plan for GraphQL API, including authentication, caching, and rate limiting"

# Infrastructure design
"Create an implementation plan for Dockerization, including development environment, production environment, and CI/CD"
```

### How Plan Mode Works

**Automatic Start**

- Starts automatically when you describe what to build
- Or just say "Create an implementation plan"

**What You Get**

- Clear requirements (user stories, success criteria)
- Design docs (architecture, data model, UI)
- Implementation steps (tasks, tracking, quality checks)
- Risk analysis and solutions

**Getting Your Approval**

- I'll show you the plan using `exit_plan_mode`
- **Important**: I always wait for your explicit OK
- I won't code without your approval
- You can request changes anytime
- TodoWrite tracking starts after you approve

### Detailed Examples

```bash
# Complex system implementation
"Create an implementation plan for an online payment system, including Stripe integration, security, and error handling"

# Frontend implementation
"Create an implementation plan for a React dashboard, including state management, component design, and testing"

# Backend implementation
"Create an implementation plan for a RESTful API, including authentication, validation, and logging"

# DevOps implementation
"Create an implementation plan for a CI/CD pipeline, including test automation, deployment, and monitoring"
```

### 3-Phase Workflow

#### Phase 1: Requirements

- **User Stories**: What are we building and why?
- **Success Criteria**: How do we know it's done?
- **Constraints**: What limits do we have?
- **Priority**: What's must-have vs nice-to-have?

#### Phase 2: Design

- **Architecture**: How will the system work?
- **Data Model**: Database schema and APIs
- **UI/UX**: Screen layouts and user flow
- **Risks**: What could go wrong and how to prevent it

#### Phase 3: Implementation

- **Task Breakdown**: Split into manageable chunks
- **Progress Tracking**: TodoWrite manages status
- **Quality Checks**: Testing and verification plan
- **Your Approval**: Show plan and wait for your OK

### Notes

**When to Use This**

- Best for complex projects
- Skip for simple fixes
- Great for 3+ step tasks or new features

**Technical Notes**

- Don't rely on `exit_plan_mode` return values
- Only your explicit approval counts
- Works differently than CLI plan mode

**Important Rules**

- Never start coding before you approve
- Always wait for your response
- Offer alternatives if something fails

### Execution Example

```bash
# Usage example
"Create an implementation plan for a user management system"

# What happens:
# 1. Plan Mode starts
# 2. Analyze requirements and pick tech
# 3. Structure the implementation
# 4. Show you the plan
# 5. Start coding after you approve
```
