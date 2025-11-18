# Refactoring Documentation

This directory contains detailed documentation for proposed refactorings to improve the pi-assistant codebase.

## Overview

The refactorings are organized by priority and estimated effort. Each document provides:
- Problem description with code examples
- Proposed solution with implementation details
- Benefits and risks
- Migration plan
- Testing strategy
- Success metrics

## Refactoring Documents

### High Priority

These refactorings provide the highest value and should be tackled first:

1. **[Audio Controller State Machine](./01-audio-controller-state-machine.md)** (3-4 days)
   - Extract 270-line monolithic function into state pattern
   - Dramatically improves testability and maintainability
   - Reduces cyclomatic complexity

2. **[Configuration Module Refactor](./02-configuration-module-refactor.md)** (2-3 days)
   - Eliminate module-level side effects
   - Enable lazy loading and testing
   - Separate concerns (loading, validation, prompting)

3. **[Error Handling Unification](./03-error-handling-unification.md)** (1-2 days)
   - Create custom exception hierarchy
   - Unified error handler utility
   - Consistent error logging across codebase

### Medium Priority

These provide significant value with moderate effort:

4. **[Device Selection Extraction](./04-device-selection-extraction.md)** (1-2 days)
   - Eliminate duplicate device selection code
   - Create shared `AudioDeviceManager`
   - Add device caching

5. **[LLM Responder Split](./05-llm-responder-split.md)** (2 days)
   - Break 88-line function into focused classes
   - Separate message building, parsing, API calls
   - Improve testability and reusability

6. **[Type Safety Improvements](./06-type-safety-improvements.md)** (2 days)
   - Add TypedDict for events and messages
   - Create Protocol definitions
   - Enable strict type checking in CI/CD

7. **[Configuration Coupling Reduction](./07-configuration-coupling-reduction.md)** (2-3 days)
   - Implement dependency injection
   - Remove direct config imports (23 modules!)
   - Enable component testing in isolation

### Low Priority (Quick Wins)

8. **[Quick Wins](./08-quick-wins.md)** (1-2 days total)
   - 8 small refactorings, each 30min-3h
   - Immediate value with minimal risk
   - Can be done incrementally

## Effort Estimates

| Priority | Items | Total Effort |
|----------|-------|--------------|
| High | 3 refactorings | 6-9 days |
| Medium | 4 refactorings | 7-9 days |
| Low | 8 quick wins | 1-2 days |
| **Total** | **15 refactorings** | **14-20 days** |

## Recommended Approach

### Phase 1: Foundation (Week 1)
Start with refactorings that unblock others:

1. **Configuration Module** (2-3 days)
   - Enables testing everywhere
   - Required by coupling reduction

2. **Error Handling** (1-2 days)
   - Used by all other refactorings
   - Quick value

3. **Quick Wins #1-4** (1 day)
   - Error logging
   - Task error handler
   - RMS calculation
   - Magic numbers

### Phase 2: Core Components (Week 2)
Tackle major architectural improvements:

4. **Audio Controller State Machine** (3-4 days)
   - Biggest complexity win
   - Improves testability dramatically

5. **Type Safety** (2 days)
   - Helps catch errors in refactoring
   - Improves development experience

### Phase 3: Refinement (Week 3)
Polish and extend:

6. **LLM Responder Split** (2 days)
7. **Device Selection Extraction** (1-2 days)
8. **Configuration Coupling Reduction** (2-3 days)
9. **Remaining Quick Wins** (1 day)

## Dependencies

Some refactorings depend on others:

```
Configuration Module Refactor
  └─→ Configuration Coupling Reduction
       └─→ All component refactorings

Error Handling Unification
  └─→ All refactorings (use custom exceptions)

Type Safety Improvements
  └─→ Helps validate all refactorings
```

## Success Metrics

Track progress with these overall metrics:

### Code Quality
- [ ] Cyclomatic complexity reduced by 40%
- [ ] Average function length < 50 lines
- [ ] No functions > 100 lines
- [ ] Test coverage > 85%

### Architecture
- [ ] No module-level side effects
- [ ] All components accept config via constructor
- [ ] Explicit dependency injection
- [ ] Clear separation of concerns

### Type Safety
- [ ] All public functions have type hints
- [ ] `mypy --strict` passes
- [ ] `pyright` strict mode passes
- [ ] Zero `type: ignore` comments (or documented)

### Error Handling
- [ ] Custom exception hierarchy in use
- [ ] No bare `except Exception` clauses
- [ ] Consistent error logging
- [ ] No silent exception swallowing

### Testing
- [ ] All components testable in isolation
- [ ] Mock/stub dependencies in tests
- [ ] Config injectable for tests
- [ ] Integration tests cover main flows

## Getting Started

1. **Read** the relevant refactoring document
2. **Create** a feature branch
3. **Follow** the migration plan
4. **Test** thoroughly (unit + integration)
5. **Review** changes against success metrics
6. **Document** any deviations or learnings

## Questions?

For questions or clarifications:
- Review the specific refactoring document
- Check related refactorings for context
- Consider starting with a Quick Win to build familiarity

## Contributing

When implementing these refactorings:

1. **One refactoring at a time** - Don't mix multiple refactorings
2. **Follow the migration plan** - Phases are designed to minimize risk
3. **Write tests first** - Ensure existing behavior is captured
4. **Update documentation** - Keep refactoring docs in sync with reality
5. **Review success metrics** - Validate the refactoring achieved its goals

## Status

| Document | Status | Started | Completed | Notes |
|----------|--------|---------|-----------|-------|
| 01-audio-controller | Planned | - | - | - |
| 02-configuration-module | Planned | - | - | - |
| 03-error-handling | Planned | - | - | - |
| 04-device-selection | Planned | - | - | - |
| 05-llm-responder | Planned | - | - | - |
| 06-type-safety | Planned | - | - | - |
| 07-configuration-coupling | Planned | - | - | - |
| 08-quick-wins | Planned | - | - | - |

---

**Last Updated**: 2025-11-18
**Total Refactorings**: 15 (3 high, 4 medium, 8 quick wins)
**Estimated Total Effort**: 14-20 days
