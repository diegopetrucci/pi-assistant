# Fix Codecov Configuration

## Problem

Codecov is requesting a coverage report from the main branch. Currently, the tests workflow only runs on pull requests, so no coverage data is being uploaded to Codecov for the main branch baseline.

## Root Cause

In `.github/workflows/tests.yml`, the workflow trigger is configured to only run on pull requests:

```yaml
on:
  pull_request:
```

This means:
- Coverage reports are only generated and uploaded during PR builds
- The main branch never generates coverage reports
- Codecov has no baseline to compare PRs against

## Solution

Update the workflow trigger to also run on pushes to the main branch.

### Required Change

In `.github/workflows/tests.yml` (lines 3-4), change:

```yaml
on:
  pull_request:
```

To:

```yaml
on:
  pull_request:
  push:
    branches:
      - main
```

## Implementation Steps

1. Edit `.github/workflows/tests.yml`
2. Update the `on:` trigger as shown above
3. Commit the change:
   ```bash
   git add .github/workflows/tests.yml
   git commit -m "Fix Codecov: add push to main trigger"
   ```
4. Push to a feature branch and create a PR
5. Once merged, the workflow will run on main and upload the first coverage report
6. Codecov will now have a baseline for future comparisons

## Expected Outcome

After merging this change:
- Tests will run on every push to main
- Coverage reports will be uploaded to Codecov from main branch builds
- Codecov will have a baseline to compare PR coverage against
- The "missing main branch report" error will be resolved

## Note

The Codecov action is already correctly configured in the workflow (line 60-63):

```yaml
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v4
  with:
    token: ${{ secrets.CODECOV_TOKEN }}
```

No changes to this section are needed.
