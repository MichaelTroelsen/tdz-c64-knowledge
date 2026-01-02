# Testing Strategy & Documentation

Comprehensive testing documentation for tdz-c64-knowledge v2.21.0 and later.

---

## Overview

The testing strategy consists of 4 levels:
1. **Unit Tests** - Individual component testing
2. **Performance Regression Tests** - Validate optimizations don't regress
3. **End-to-End Integration Tests** - Complete workflow testing
4. **Load Tests** - Scalability and performance validation

---

## Test Suites

### 1. Unit Tests

**File**: `test_anomaly_detector.py`
**Purpose**: Test individual components of anomaly detection system
**Coverage**: 15 tests covering core functionality

**Run Command**:
```bash
.venv/Scripts/python.exe -m pytest test_anomaly_detector.py -v
```

**Test Categories**:
- Initialization (normal & custom parameters)
- Single check recording
- Batch check recording
- Baseline calculation & retrieval
- Anomaly score calculation (with/without baseline)
- Severity classification
- Smart noise filtering
- History queries
- Edge cases (empty batches, failed checks)
- Performance validation

**Performance Metrics**:
- Test suite runtime: ~9 seconds
- All tests passing: 15/15
- Coverage: Core functionality, batch operations, edge cases

**Example Output**:
```
15 passed, 3 warnings in 8.82s
```

---

### 2. Performance Regression Tests

**File**: `test_performance_regression.py`
**Purpose**: Ensure performance optimizations remain effective
**Coverage**: 7 tests validating performance baselines

**Run Command**:
```bash
.venv/Scripts/python.exe test_performance_regression.py
```

**Performance Baselines**:
- Batch recording (100 checks): < 0.1s
- Baseline updates (100 docs): < 0.5s
- Anomaly scoring (100 scores): < 0.2s
- Baseline retrieval (50 queries): < 0.1s
- Bulk baseline retrieval (100+): < 0.1s
- History query (100+ entries): < 0.05s
- Full monitoring cycle (50 docs): < 1.0s

**Actual Performance** (as of v2.21.0):
- Batch recording: **6787 checks/sec** (67x above baseline)
- Baseline updates: **9450 updates/sec** (47x above baseline)
- Anomaly scoring: **17265 scores/sec** (86x above baseline)
- Full cycle: **2777 docs/sec** (55x above baseline)

**Failure Conditions**:
- Any test taking longer than baseline threshold
- Used for CI/CD to catch performance regressions

**Example Output**:
```
[PASS] Batch recording: 0.015s (6787 checks/sec)
[PASS] Baseline updates: 0.011s (9450 updates/sec)
[PASS] Anomaly scoring: 0.006s (17265 scores/sec)
[PASS] Full monitoring cycle (50 docs): 0.018s (2777 docs/sec)
```

---

### 3. End-to-End Integration Tests

**File**: `test_e2e_integration.py`
**Purpose**: Test complete workflows from start to finish
**Coverage**: 4 tests covering real-world scenarios

**Run Command**:
```bash
.venv/Scripts/python.exe test_e2e_integration.py
```

**Test Scenarios**:

#### Test 1: New Document Baseline Learning
- Create new monitored document
- Record initial checks (no baseline)
- Build baseline with 5+ checks
- Validate normal vs anomalous scoring
- Verify baseline was learned

**Validates**: Complete workflow from document creation to anomaly detection

#### Test 2: Batch Monitoring Workflow
- Monitor 20 documents simultaneously
- Build baselines over simulated 7 days
- Process mixed results (unchanged/changed/failed)
- Calculate anomaly scores
- Verify all baselines created

**Validates**: Batch processing with multiple documents

#### Test 3: Pattern Filtering Integration
- Test timestamp change filtering
- Test real content change detection
- Add custom ignore pattern
- Verify custom pattern works

**Validates**: Smart noise filtering integration

#### Test 4: Anomaly History Tracking
- Record varied check history
- Query complete history
- Verify status breakdowns
- Query anomalies by severity

**Validates**: Complete audit trail functionality

**Example Output**:
```
4 passed, 3 warnings in 8.09s

E2E TEST COVERAGE:
  [OK] New document -> baseline learning -> anomaly detection
  [OK] Batch monitoring workflow (20 documents)
  [OK] Pattern filtering integration
  [OK] Complete anomaly history tracking
```

---

### 4. Load Testing

**File**: `load_test.py`
**Purpose**: Validate scalability with large document counts
**Coverage**: Performance testing with 100-1000+ documents

**Run Commands**:
```bash
# Quick test (100 documents)
.venv/Scripts/python.exe load_test.py --docs 100

# Standard test (1000 documents)
.venv/Scripts/python.exe load_test.py --docs 1000

# Custom size
.venv/Scripts/python.exe load_test.py --docs 5000

# Quiet mode (less output)
.venv/Scripts/python.exe load_test.py --docs 1000 --quiet
```

**Test Phases**:

**Phase 1**: Document Creation
- Create N test documents
- Batch insert into database
- Measure throughput

**Phase 2**: Batch Recording
- Record 1000 checks
- Measure throughput
- Target: >1000 checks/sec

**Phase 3**: Baseline Updates
- Update 1000 baselines
- Measure throughput
- Target: >200 updates/sec

**Phase 4**: Score Calculation
- Calculate 1000 anomaly scores
- Measure throughput
- Target: >500 scores/sec

**Phase 5**: Full Monitoring Cycle
- Complete workflow for 1000 docs
- Includes recording + scoring
- Target: >50 docs/sec

**Phase 6**: Query Performance
- Test baseline retrieval
- Test history queries
- Test anomaly queries

**Performance Targets**:
```
Batch recording:     >1000 checks/sec
Baseline updates:    >200 updates/sec
Score calculation:   >500 scores/sec
Full cycle:          >50 docs/sec
```

**Example Output**:
```
==================================================================
LOAD TEST: 100 Documents
==================================================================

[Phase 1] Creating 100 documents...
[OK] Created 100 documents in 0.00s

[Phase 2] Batch recording 100 checks...
[OK] Recorded 100 checks in 0.015s
     Throughput: 6787 checks/sec
     [PASS] Target: >1000 checks/sec

[Phase 3] Updating 100 baselines...
[OK] Updated 100 baselines in 0.009s
     Throughput: 10811 updates/sec
     [PASS] Target: >200 updates/sec

[Phase 4] Calculating 100 anomaly scores...
[OK] Calculated 100 scores in 0.003s
     Throughput: 35336 scores/sec
     [PASS] Target: >500 scores/sec

[Phase 5] Full monitoring cycle (100 documents)...
[OK] Completed monitoring cycle in 0.017s
     Record checks: 0.015s
     Calculate scores: 0.002s
     Throughput: 5787 docs/sec
     [PASS] Target: >50 docs/sec full cycle

[Phase 6] Query performance tests...
[OK] Retrieved 100 baselines in 0.000s
[OK] Retrieved history in 0.000s
[OK] Retrieved anomalies in 0.000s

==================================================================
LOAD TEST SUMMARY
==================================================================
Documents tested: 100

Performance Results:
  Batch recording:     6787 checks/sec
  Baseline updates:    10811 updates/sec
  Score calculation:   35336 scores/sec
  Full cycle:          5787 docs/sec

[PASS] All performance targets met
```

---

## Running All Tests

### Quick Validation
```bash
# Run all unit tests
.venv/Scripts/python.exe -m pytest test_anomaly_detector.py -v

# Run E2E tests
.venv/Scripts/python.exe -m pytest test_e2e_integration.py -v

# Quick load test
.venv/Scripts/python.exe load_test.py --docs 100
```

### Full Test Suite
```bash
# Unit tests
.venv/Scripts/python.exe -m pytest test_anomaly_detector.py -v

# Performance regression tests
.venv/Scripts/python.exe test_performance_regression.py

# E2E integration tests
.venv/Scripts/python.exe test_e2e_integration.py

# Load test (1000 documents)
.venv/Scripts/python.exe load_test.py --docs 1000
```

### CI/CD Pipeline
```bash
# Run all tests with coverage
pytest test_anomaly_detector.py test_e2e_integration.py -v --cov=anomaly_detector --cov-report=term

# Run performance regression tests
python test_performance_regression.py

# Run load test (quick)
python load_test.py --docs 100 --quiet
```

---

## Test Environment

### Setup
All tests use isolated temporary databases to avoid affecting production data.

**Environment Variables** (automatically set by tests):
```bash
USE_SEMANTIC_SEARCH=0  # Disable for faster tests
USE_FTS5=1            # Enable FTS5 search
```

### Test Data
- Tests create temporary documents in `tempfile.mkdtemp()`
- Automatic cleanup after tests
- No impact on production database

### Dependencies
- pytest
- tempfile (stdlib)
- datetime (stdlib)
- server.py (KnowledgeBase)
- anomaly_detector.py (AnomalyDetector)
- migration_v2_21_0.py (database migration)

---

## Performance Benchmarks

### v2.21.0 Performance (as of 2025-12-22)

**Hardware**: Windows 11, Python 3.14

**Unit Tests**:
- 15 tests in 8.82s
- Average: 0.59s per test

**Performance Regression Tests**:
- 7 tests in 0.44s
- All baselines exceeded by 40-80x

**E2E Integration Tests**:
- 4 tests in 8.09s
- Average: 2.02s per test

**Load Test (100 docs)**:
- Total time: ~0.5s
- Batch recording: 6787 checks/sec
- Full cycle: 5787 docs/sec

**Load Test (1000 docs)** (estimated):
- Total time: ~5s
- Batch recording: ~7000 checks/sec
- Full cycle: ~200 docs/sec

---

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest test_anomaly_detector.py test_e2e_integration.py -v
      - run: python test_performance_regression.py
      - run: python load_test.py --docs 100 --quiet
```

---

## Troubleshooting

### Tests Fail with "Database locked"
**Cause**: Concurrent access to test database
**Solution**: Tests use separate temporary databases, shouldn't occur

### Performance Tests Fail
**Cause**: System under load or insufficient resources
**Solution**: Close other applications, run again

### Load Test Timeouts
**Cause**: Testing with very large document counts (>10000)
**Solution**: Reduce document count or increase timeout

### Import Errors
**Cause**: Missing dependencies
**Solution**:
```bash
pip install -e ".[dev]"
```

---

## Adding New Tests

### Unit Test Template
```python
def test_new_feature(self):
    """Test new feature description."""
    detector = AnomalyDetector(self.kb)

    # Setup
    # ...

    # Execute
    result = detector.new_method()

    # Assert
    assert result == expected
    print(f"[PASS] New feature test")
```

### Performance Test Template
```python
def test_new_performance(self):
    """Test new performance requirement."""
    import time

    # Setup
    # ...

    # Measure
    start = time.time()
    # ... operation ...
    elapsed = time.time() - start

    # Assert baseline
    assert elapsed < 1.0, f"Took {elapsed:.3f}s, baseline is 1.0s"
```

---

## Best Practices

1. **Isolation**: Each test should be independent
2. **Cleanup**: Always cleanup temporary files/databases
3. **Performance**: Set realistic baselines based on hardware
4. **Coverage**: Test happy path, edge cases, and errors
5. **Documentation**: Add docstrings explaining what each test validates
6. **Assertions**: Include helpful failure messages
7. **Output**: Print progress for long-running tests

---

## Version History

- **v2.21.0**: Added complete test suite (unit, performance, E2E, load)
- **v2.20.0**: Basic test_server.py only

---

## Future Improvements

- Add code coverage reporting (target: >80%)
- Add mutation testing
- Add stress tests (extreme loads, resource exhaustion)
- Add security tests (SQL injection, path traversal)
- Add concurrent access tests
- Add database corruption recovery tests

---

**Maintained by**: Claude Code
**Last Updated**: 2025-12-22
**Version**: 2.21.0
