## Code Explain

Explains how code works in detail.

### Usage

```bash
# Show a file and ask for explanation
cat <file>
"Explain how this code works"
```

### Basic Examples

```bash
# Understand Rust ownership
cat main.rs
"Explain the ownership and lifetimes in this Rust code"

# Explain an algorithm
grep -A 50 "quicksort" sort.rs
"How does this sorting work? What's its time complexity?"

# Explain design patterns
cat factory.rs
"What design pattern is this? What are the benefits?"
```

### Collaboration with Claude

```bash
# Beginner-friendly explanation
cat complex_function.py
"Explain this code line by line for someone new to programming"

# Performance check
cat algorithm.rs
"Find performance problems and how to fix them"

# Visual explanation
cat state_machine.js
"Show me the flow with ASCII diagrams"

# Security check
cat auth_handler.go
"What security issues do you see?"
```

### Detailed Examples

```bash
# Complex logic breakdown
cat recursive_parser.rs
"Break down this recursive parser:
1. How does it flow?
2. What does each function do?
3. How are edge cases handled?
4. What could be better?"

# Async code explanation
cat async_handler.ts
"Explain this async code:
1. How do the Promises flow?
2. How are errors handled?
3. What runs in parallel?
4. Could this deadlock?"

# Architecture overview
ls -la src/ && cat src/main.rs src/lib.rs
"Explain how this project is structured"
```

### What You'll Get

Not just what the code does, but also:

- Why it's written that way
- What benefits it provides
- What problems might come up
