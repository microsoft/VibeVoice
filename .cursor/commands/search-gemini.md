## Gemini Web Search

Execute web searches via Gemini CLI to obtain the latest information.

### Usage

```bash
# Web search via Gemini CLI (required)
gemini --prompt "WebSearch: <search_query>"
```

### Basic Examples

```bash
# Using Gemini CLI
gemini --prompt "WebSearch: React 19 new features"
gemini --prompt "WebSearch: TypeError Cannot read property of undefined solution"
```

### Collaboration with Claude

```bash
# Document search and summarization
gemini --prompt "WebSearch: Next.js 14 App Router official documentation"
"Summarize the search results and explain the main features"

# Error investigation
cat error.log
gemini --prompt "WebSearch: [error_message] solution"
"Propose the most appropriate solution from the search results"

# Technology comparison
gemini --prompt "WebSearch: Rust vs Go performance benchmark 2024"
"Summarize the performance differences from the search results"
```

### Detailed Examples

```bash
# Information gathering from multiple sources
gemini --prompt "WebSearch: GraphQL best practices 2024 multiple sources"
"Summarize information from multiple reliable sources in the search results"

# Investigating changes over time
gemini --prompt "WebSearch: JavaScript ES2015 ES2016 ES2017 ES2018 ES2019 ES2020 ES2021 ES2022 ES2023 ES2024 features"
"Summarize the main changes in each version in chronological order"

# Search limited to specific domain
gemini --prompt "WebSearch: site:github.com Rust WebAssembly projects stars:>1000"
"List the top 10 projects by number of stars"

# Latest security information
gemini --prompt "WebSearch: CVE-2024 Node.js vulnerabilities"
"Summarize the impact and countermeasures of found vulnerabilities"
```

### Prohibited Items

- **Prohibited to use Claude's built-in WebSearch tool**
- When web search is needed, always use `gemini --prompt "WebSearch: ..."`

### Important Notes

- **When Gemini CLI is available, always use `gemini --prompt "WebSearch: ..."`**
- Web search results are not always the latest
- It is recommended to verify important information with official documentation or reliable sources
