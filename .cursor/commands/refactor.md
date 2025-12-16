## Refactor

Cleans up code safely step-by-step and checks SOLID principles.

### Usage

```bash
# Find complex code to refactor
find . -name "*.js" -exec wc -l {} + | sort -rn | head -10
"Make these big files simpler"

# Find duplicate code
grep -r "function processUser" . --include="*.js"
"Combine these duplicate functions"

# Check SOLID principles
grep -r "class.*Service" . --include="*.js" | head -10
"Do these classes have just one job?"
```

### Basic Examples

```bash
# Find long methods
grep -A 50 "function" src/*.js | grep -B 50 -A 50 "return" | wc -l
"Break up methods over 50 lines"

# Find complex conditions
grep -r "if.*if.*if" . --include="*.js"
"Simplify these nested ifs"

# Find code smells
grep -r "TODO\|FIXME\|HACK" . --exclude-dir=node_modules
"Fix these TODO comments"
```

### Refactoring Techniques

#### Extract Method (Split Big Functions)

```javascript
// Before: Long method
function processOrder(order) {
  // 50 lines of complex processing
}

// After: Separation of responsibilities
function processOrder(order) {
  validateOrder(order);
  calculateTotal(order);
  saveOrder(order);
}
```

#### Replace Conditional with Polymorphism (Remove Switch/If Chains)

```javascript
// Before: switch statement
function getPrice(user) {
  switch (user.type) {
    case 'premium': return basePrice * 0.8;
    case 'regular': return basePrice;
  }
}

// After: Strategy pattern
class PremiumPricing {
  calculate(basePrice) { return basePrice * 0.8; }
}
```

### SOLID Principles We Check

```
S - Single Responsibility
├─ One class = one job
├─ Only one reason to change
└─ Clear boundaries

O - Open/Closed
├─ Easy to extend
├─ Don't modify existing code
└─ Add features without breaking things

L - Liskov Substitution
├─ Child classes can replace parents
├─ Keep the same behavior
└─ Don't break expectations

I - Interface Segregation
├─ Small, focused interfaces
├─ No unused methods
└─ Each interface for a specific role

D - Dependency Inversion
├─ Depend on interfaces, not classes
├─ Keep implementations separate
└─ Use dependency injection
```

### How to Refactor Safely

1. **Check the current state**
   - Measure complexity
   - Find duplicate code
   - Map dependencies

2. **Make small changes**
   - 15-30 minute chunks
   - Test after each change
   - Commit often

3. **Verify quality**
   - Keep tests passing
   - Check performance
   - Get code reviewed

### Code Smells to Fix

- **God Object**: Classes that do everything
- **Long Method**: Functions over 50 lines
- **Duplicate Code**: Same logic in multiple places
- **Large Class**: Classes over 300 lines
- **Long Parameter List**: More than 3 parameters

### Automation Support

```bash
# Static analysis
npx complexity-report src/
sonar-scanner

# Code formatting
npm run lint:fix
prettier --write src/

# Test execution
npm test
npm run test:coverage
```

### Important Rules

- **Don't break things**: Keep the same behavior
- **Test first**: Write tests before changing code
- **Small steps**: No huge changes all at once
- **Test often**: Run tests after every change
