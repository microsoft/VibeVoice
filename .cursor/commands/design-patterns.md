## Design Patterns

Suggests design patterns for your code and checks if it follows SOLID principles.

### Usage

```bash
/design-patterns [analysis_target] [options]
```

### Options

- `--suggest`: Suggest applicable patterns (default)
- `--analyze`: Analyze existing pattern usage
- `--refactor`: Generate refactoring proposals
- `--solid`: Check compliance with SOLID principles
- `--anti-patterns`: Detect anti-patterns

### Basic Examples

```bash
# Analyze patterns for entire project
/design-patterns

# Suggest patterns for specific file
/design-patterns src/services/user.js --suggest

# Check SOLID principles
/design-patterns --solid

# Detect anti-patterns
/design-patterns --anti-patterns
```

### Pattern Categories

#### 1. Creational Patterns

- **Factory Pattern**: Abstracts object creation
- **Builder Pattern**: Step-by-step construction of complex objects
- **Singleton Pattern**: Ensures only one instance exists
- **Prototype Pattern**: Creates object clones

#### 2. Structural Patterns

- **Adapter Pattern**: Converts interfaces
- **Decorator Pattern**: Dynamically adds functionality
- **Facade Pattern**: Simplifies complex subsystems
- **Proxy Pattern**: Controls access to objects

#### 3. Behavioral Patterns

- **Observer Pattern**: Implements event notifications
- **Strategy Pattern**: Switches algorithms
- **Command Pattern**: Encapsulates operations
- **Iterator Pattern**: Traverses collections

### SOLID Principles We Check

```
S - Single Responsibility (one class, one job)
O - Open/Closed (open for extension, closed for modification)
L - Liskov Substitution (subtypes should be replaceable)
I - Interface Segregation (don't force unused methods)
D - Dependency Inversion (depend on abstractions, not details)
```

### Output Example

```
Design Pattern Analysis Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Currently Used Patterns
├─ Observer Pattern: EventEmitter (12 instances)
├─ Factory Pattern: UserFactory (3 instances)
├─ Singleton Pattern: DatabaseConnection (1 instance)
└─ Strategy Pattern: PaymentProcessor (5 instances)

Recommended Patterns
├─ [HIGH] Repository Pattern
│  └─ Where: src/models/*.js
│  └─ Why: Separate data access from business logic
│  └─ Example:
│      class UserRepository {
│        async findById(id) { ... }
│        async save(user) { ... }
│      }
│
├─ [MED] Command Pattern
│  └─ Where: src/api/handlers/*.js
│  └─ Why: Standardize how requests are handled
│
└─ [LOW] Decorator Pattern
   └─ Where: src/middleware/*.js
   └─ Why: Better way to combine features

SOLID Violations Found
├─ [S] UserService: Does too much (auth AND authorization)
├─ [O] PaymentGateway: Must change code to add payment types
├─ [D] EmailService: Depends on specific classes, not interfaces
└─ [I] IDataStore: Has methods nobody uses

How to Fix
1. Split UserService into AuthService and AuthorizationService
2. Add a PaymentStrategy interface for new payment types
3. Create an EmailService interface
4. Break up IDataStore into smaller interfaces
```

### Advanced Usage Examples

```bash
# See what happens if you use a pattern
/design-patterns --impact-analysis Repository

# Get example code for a pattern
/design-patterns --generate Factory --for src/models/Product.js

# Find patterns that work well together
/design-patterns --combine --context "API with caching"

# Check your architecture
/design-patterns --architecture MVC
```

### Example: Before and After

#### Before (Problem Code)

```javascript
class OrderService {
  processOrder(order, paymentType) {
    if (paymentType === "credit") {
      // Credit card processing
    } else if (paymentType === "paypal") {
      // PayPal processing
    }
    // Other payment methods...
  }
}
```

#### After (Applying Strategy Pattern)

```javascript
// Strategy interface
class PaymentStrategy {
  process(amount) {
    throw new Error("Must implement process method");
  }
}

// Concrete strategies
class CreditCardPayment extends PaymentStrategy {
  process(amount) {
    /* Implementation */
  }
}

// Context
class OrderService {
  constructor(paymentStrategy) {
    this.paymentStrategy = paymentStrategy;
  }

  processOrder(order) {
    this.paymentStrategy.process(order.total);
  }
}
```

### Anti-Patterns We Find

- **God Object**: Classes that do everything
- **Spaghetti Code**: Tangled mess of control flow
- **Copy-Paste Programming**: Same code everywhere
- **Magic Numbers**: Random numbers with no explanation
- **Callback Hell**: Callbacks inside callbacks inside callbacks

### Best Practices

1. **Go slow**: Add patterns one at a time
2. **Need first**: Only use patterns to solve real problems
3. **Talk it out**: Get team buy-in before big changes
4. **Write it down**: Document why you chose each pattern
