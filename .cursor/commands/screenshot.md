## Screenshot

Capture screenshots on macOS and analyze the images.

### Usage

```bash
/screenshot [options]
```

### Options

- None: Select window (Claude will confirm options)
- `--window`: Capture a specific window
- `--full`: Capture the entire screen
- `--crop`: Select a region to capture

### Basic Examples

```bash
# Capture and analyze a window
/screenshot --window
"Analyze the captured screen"

# Select a region and analyze
/screenshot --crop
"Explain the content of the selected region"

# Capture full screen and analyze
/screenshot --full
"Analyze the overall screen composition"
```

### Collaboration with Claude

```bash
# No specific problem - situation analysis
/screenshot --crop
(Claude will automatically analyze screen content, explaining elements and composition)

# UI/UX problem analysis
/screenshot --window
"Propose problems and improvements for this UI"

# Error analysis
/screenshot --window
"Tell me the cause and solution for this error message"

# Design review
/screenshot --full
"Evaluate this design from a UX perspective"

# Code analysis
/screenshot --crop
"Point out problems in this code"

# Data visualization analysis
/screenshot --crop
"Analyze trends visible in this graph"
```

### Detailed Examples

```bash
# Analysis from multiple perspectives
/screenshot --window
"Analyze this screen regarding:
1. UI consistency
2. Accessibility issues
3. Improvement proposals"

# Multiple captures for comparative analysis
/screenshot --window
# (Save before image)
# Make changes
/screenshot --window
# (Save after image)
"Compare before and after images, analyzing changes and improvement effects"

# Focus on specific elements
/screenshot --crop
"Evaluate whether the selected button design harmonizes with other elements"
```

### Prohibited Items

- **Prohibited to say "captured" when no screenshot was taken**
- **Prohibited to attempt analysis of non-existent image files**
- **The `/screenshot` command does not actually capture screenshots**

### Notes

- If no option is specified, please present the following choices:

  ```
  "How would you like to capture the screenshot?
  1. Select window (--window) → screencapture -W
  2. Full screen (--full) → screencapture -x
  3. Select region (--crop) → screencapture -i"
  ```

- Start image analysis after the user has executed the screencapture command
- Specifying specific problems or perspectives enables more focused analysis
