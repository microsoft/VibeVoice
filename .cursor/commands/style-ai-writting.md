## AI Writing Check

Detects mechanical patterns in AI-generated text and provides suggestions for improving to more natural Japanese.

### Usage

```bash
/ai-writing-check [options]
```

### Options

- None: Analyze current file or selected text
- `--file <path>`: Analyze specific file
- `--dir <path>`: Batch analyze files in directory
- `--severity <level>`: Detection level (all/high/medium)
- `--fix`: Automatically fix detected patterns

### Basic Examples

```bash
# Check AI writing style in file
cat README.md
/ai-writing-check
"Check this document for AI writing style and suggest improvements"

# Analyze specific file
/ai-writing-check --file docs/guide.md
"Detect AI-like expressions and suggest corrections to natural expressions"

# Scan entire project
/ai-writing-check --dir . --severity high
"Report only critical AI writing issues in the project"
```

### Detected Patterns

#### 1. Mechanical List Format Patterns

```markdown
Examples detected:

- **Important**: This is an important item
- Completed item (with checkmark emoji)
- Hot topic (with fire emoji)
- Ready to start (with rocket emoji)

Improved examples:

- Important item: This is an important item
- Completed item
- Notable topic
- Ready to start
```

#### 2. Exaggerated/Hype Expressions

```markdown
Examples detected:
Revolutionary technology will change the industry.
This completely solves the problem.
Works like magic.

Improved examples:
Effective technology brings change to the industry.
Solves many problems.
Works smoothly.
```

#### 3. Mechanical Emphasis Patterns

```markdown
Examples detected:
**Idea**: New proposal (with lightbulb emoji)
**Caution**: Important warning (with warning emoji)

Improved examples:
Idea: New proposal
Note: Important warning
```

#### 4. Redundant Technical Writing

```markdown
Examples detected:
First, we will perform the setup.
You can use this tool.
Performance is greatly improved.

Improved examples:
First, perform setup.
You can use this tool.
Performance improves by 30%.
```

### Collaboration with Claude

```bash
# Analyze entire document for AI writing style
cat article.md
/ai-writing-check
"Analyze and suggest improvements from these perspectives:
1. Detection of mechanical expressions
2. Suggestions for correction to natural Japanese
3. Priority-based improvement list"

# Focus on specific patterns
/ai-writing-check --file blog.md
"Pay special attention to exaggerated and redundant expressions and suggest improvements"

# Batch check multiple files
find . -name "*.md" -type f
/ai-writing-check --dir docs/
"Analyze AI writing style throughout the documentation and create a summary"
```

### Detailed Examples

```bash
# Compare before and after improvement
/ai-writing-check --file draft.md
"Detect AI-like expressions and present them in the following format:
- Problem areas (with line numbers)
- Type of problem and reason
- Specific improvement suggestions
- Effect of improvement"

# Auto-fix mode
/ai-writing-check --file report.md --fix
"Automatically fix detected patterns and report results"

# Project AI writing style report
/ai-writing-check --dir . --severity all
"Analyze AI writing style throughout the project and provide:
1. Statistical information (detection count by pattern)
2. Top 5 most problematic files
3. Improvement priority matrix
4. Step-by-step improvement plan"
```

### Advanced Usage Examples

```bash
# Apply custom rules
/ai-writing-check --file spec.md
"Check technical specifications with these additional criteria:
- Ambiguous expressions (appropriate, as needed)
- Lack of specificity (fast → specific numbers)
- Inconsistent terminology usage"

# Check for CI/CD integration
/ai-writing-check --dir docs/ --severity high
"Output results in GitHub Actions executable format:
- Number of errors and filenames
- Line numbers requiring correction
- Exit code configuration"

# Style guide compliance check
/ai-writing-check --file manual.md
"Additional checks based on company style guide:
- Honorific usage (unification of desu/masu form)
- Appropriate use of technical terms
- Consideration for readers"
```

### Notes

- AI writing style determination varies by context, so treat suggestions as reference
- Adjust criteria according to document type (technical documents, blogs, manuals, etc.)
- You don't need to accept all suggestions; select appropriate ones
- The `--fix` option automatically corrects detected patterns

### Command Execution Behavior

When you run the `/ai-writing-check` command, Claude performs the following processes:

1. **Pattern Detection**: Detects AI-like patterns from specified files or text
2. **Specific Correction Suggestions**: Presents correction suggestions with line numbers for each issue
3. **--fix Mode**: Automatically fixes detected patterns and displays a summary of results
4. **Report Generation**: Provides detection count, improvement priority, and comparison before/after correction

Claude reads the actual file contents and performs analysis based on the rules of textlint-rule-preset-ai-writing.

### Reference

This command is created with reference to the [textlint-rule-preset-ai-writing](https://github.com/textlint-ja/textlint-rule-preset-ai-writing) rule set. It is a textlint rule preset for detecting mechanical patterns in AI-generated text and promoting more natural expressions.
