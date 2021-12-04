# DCBLSTM Tests

All tests in the project were ran using Python's unittest testing framework (https://docs.python.org/3/library/unittest.html).

**Run all unittests from main project directory:**
```
python3 -m unittest discover
```

**To run tests for specific module, from the main project directory run:**
```
python -m unittest tests.MODULE_NAME -v
```

You can add the flag *-b* to suppress some of the verbose output when running the unittests.

Unit tests
----------

* `test_datasets.py` - tests for training and testing datasets.
* `test_evaluate.py` - tests for all model evaluation functions.
* `test_models.py` - tests for all models used in project.
* `test_utils.py` - tests for all utility functions.
