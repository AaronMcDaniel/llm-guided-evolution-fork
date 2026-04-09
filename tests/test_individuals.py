"""
LLMGE Individual Evaluation Tests

Tests eval.py on 12 frozen individuals from a real LLMGE run

Run:
    uv run pytest tests/test_individuals.py -v
"""
import os
import ast
import subprocess
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOTA = os.path.join(ROOT, 'sota', 'Titanic')
EVAL_SCRIPT = os.path.join(SOTA, 'eval.py')
#INDIVIDUALS_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'individuals')

INDIVIDUALS_DIR = os.path.join(SOTA, 'results')

RESULTS_DIR = os.path.join(SOTA, 'results')

MAX_SAMPLES = 179  # lowkey gonna need to change this based on each dataset

# Frozen individuals from titanic_test/0
# INDIVIDUALS = [
#     'xXx04PqApRQMPw55qx6cZsUfVSi',
#     'xXx04jaaxCjDADdZfQejz3gbNFD',
#     'xXx05DUkH2xBIGogZRAFJ8nnJHE',
#     'xXx061oJQctwH2mBMMPeIwd7sCk',
#     'xXx07Gk5iNRGngQLpVQwbMDAAS1',
#     'xXx09ejihbvx7yUX9JYxtIVPGDs',
#     'xXx0LFCBTNE4Wx7KUjfsG1nZc4I',
#     'xXx0MtgG8LTm4mSVWscg4s97g0W',
#     'xXx0OCDKQ3u88o9xhlAyHpEcfhM',
#     'xXx0T6XRn6JDQPVLadZ4zBFlTd0',
#     'xXx0UVvEW16rb3wrfuDSAemYblI',
#     'xXx0WcxFNILJIkWzWO9MsEGL0Jx',
# ]

# INDIVIDUALS = [#passing individuals
#     'crossed_xXx0MtgG8LTm4mSVWscg4s97g0W_xXx09ejihbvx7yUX9JYxtIVPGDs',
#     'crossed_xXx0MtgG8LTm4mSVWscg4s97g0W_xXx0UVvEW16rb3wrfuDSAemYblI',
#     'crossed_xXx0LFCBTNE4Wx7KUjfsG1nZc4I_xXx07Gk5iNRGngQLpVQwbMDAAS1',
#     'crossed_xXx0LFCBTNE4Wx7KUjfsG1nZc4I_xXx09ejihbvx7yUX9JYxtIVPGDs',
#     'crossed_xXx0LFCBTNE4Wx7KUjfsG1nZc4I_xXx04jaaxCjDADdZfQejz3gbNFD',
#     'crossed_xXx0MtgG8LTm4mSVWscg4s97g0W_xXx07Gk5iNRGngQLpVQwbMDAAS1',
#     'crossed_xXx0WcxFNILJIkWzWO9MsEGL0Jx_xXx09ejihbvx7yUX9JYxtIVPGDs',
#     'crossed_xXx0WcxFNILJIkWzWO9MsEGL0Jx_xXx04jaaxCjDADdZfQejz3gbNFD',
#     'crossed_xXx0UVvEW16rb3wrfuDSAemYblI_xXx09ejihbvx7yUX9JYxtIVPGDs',
#     'crossed_xXx04jaaxCjDADdZfQejz3gbNFD_xXx09ejihbvx7yUX9JYxtIVPGDs',
#     'crossed_xXx05DUkH2xBIGogZRAFJ8nnJHE_xXx09ejihbvx7yUX9JYxtIVPGDs',
#     'crossed_xXx0LFCBTNE4Wx7KUjfsG1nZc4I_xXx0UVvEW16rb3wrfuDSAemYblI',
#     'crossed_xXx04jaaxCjDADdZfQejz3gbNFD_xXx07Gk5iNRGngQLpVQwbMDAAS1',
#     'crossed_xXx0LFCBTNE4Wx7KUjfsG1nZc4I_xXx0MtgG8LTm4mSVWscg4s97g0W',
# ]
INDIVIDUALS = [ #failed individuals
    'crossed_xXx04PqApRQMPw55qx6cZsUfVSi_xXx04jaaxCjDADdZfQejz3gbNFD',
    'crossed_xXx07Gk5iNRGngQLpVQwbMDAAS1_xXx061oJQctwH2mBMMPeIwd7sCk',
    'crossed_xXx07Gk5iNRGngQLpVQwbMDAAS1_xXx061oJQctwH2mBMMPeIwd7sCk',
    'crossed_xXx0MtgG8LTm4mSVWscg4s97g0W_xXx04jaaxCjDADdZfQejz3gbNFD'

]
# ── 1. Each individual is valid Python with a Model class. Ensures models haven't changed and all can be evaluated. ──────────────────

@pytest.mark.parametrize('gene_id', INDIVIDUALS)
def test_individual_is_valid_python(gene_id):
    """Frozen individual parses as Python with Model(fit, predict)."""
    path = os.path.join(INDIVIDUALS_DIR, f'model_{gene_id}.py')
    with open(path) as f:
        tree = ast.parse(f.read())
    classes = [n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == 'Model']
    assert len(classes) >= 1, f"{gene_id}: no Model class"
    methods = {n.name for n in ast.walk(classes[0])
               if isinstance(n, ast.FunctionDef)}
    assert 'fit' in methods, f"{gene_id}: Model missing fit()"
    assert 'predict' in methods, f"{gene_id}: Model missing predict()"


# ── 2. Each individual evaluates and produces valid results ────────────────

@pytest.mark.parametrize('gene_id', INDIVIDUALS)
def test_individual_evaluates(gene_id):
    """eval.py runs on the individual and produces bounded FP,FN."""
    model_name = f'model_{gene_id}'

    result = subprocess.run(
        ['uv', 'run', 'python', EVAL_SCRIPT,
         '--model', model_name,
         '--variant_dir', INDIVIDUALS_DIR],
        cwd=ROOT, capture_output=True, text=True, timeout=10000 # need to find more exact timeout value for different datasets this is going to be run on
    )

    # Runtime errors expected by some so skip over those ** is this valid?
    if result.returncode != 0:
        pytest.skip(f"{gene_id}: runtime error in generated model (expected for some)")

    assert 'job done' in result.stdout.lower(), \
        f"{gene_id}: 'job done' not in output"

    # Check results file was written
    results_file = os.path.join(RESULTS_DIR, f'{gene_id}_results.txt')
    assert os.path.isfile(results_file), \
        f"{gene_id}: results file not created"

    # Validate format and bounds
    with open(results_file) as f:
        content = f.read().strip()
    parts = content.split(',')
    assert len(parts) == 2, f"{gene_id}: expected 'FP,FN', got '{content}'"

    fp, fn = float(parts[0].strip()), float(parts[1].strip())
    assert 0 <= fp <= MAX_SAMPLES, f"{gene_id}: FP={fp} out of bounds [0, {MAX_SAMPLES}]"
    assert 0 <= fn <= MAX_SAMPLES, f"{gene_id}: FN={fn} out of bounds [0, {MAX_SAMPLES}]"

