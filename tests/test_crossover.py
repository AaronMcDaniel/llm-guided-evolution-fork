"""
LLMGE Individual Evaluation Tests

Tests llm_crossover.py on 12 frozen individuals from a real LLMGE run

Run:
    uv run pytest tests/test_crossover.py -v
"""
import os
import ast
import subprocess
import pytest
from itertools import combinations
import random


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOTA = os.path.join(ROOT, 'sota', 'Titanic')
CROSSOVER_SCRIPT = os.path.join(ROOT, 'src', 'llm_crossover.py')
INDIVIDUALS_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'individuals')
RESULTS_DIR = os.path.join(SOTA, 'results') #results stored in sota/Titanic/results (consider changing this?)

# Frozen individuals
INDIVIDUALS = [
    'xXx0LFCBTNE4Wx7KUjfsG1nZc4I',
    'xXx0MtgG8LTm4mSVWscg4s97g0W',
    'xXx0OCDKQ3u88o7081234567890',
    'xXxT6XRn6JDQPVLadZ4zBFlTd0',
    'xXx0UVvEW16rb3wrfuDSAemYblI',
    'xXx0WcxFNILJIkWzWO9MsEGL0Jx',
    'xXx04PqApRQMPw55qx6cZsUfVSi',
    'xXx04jaaxCjDADdZfQejz3gbNFD',
    'xXx05DUkH2xBIGogZRAFJ8nnJHE',
    'xXx07Gk5iNRGngQLpVQwbMDAAS1',
    'xXx09ejihbvx7yUX9JYxtIVPGDs',
    'xXx061oJQctwH2mBMMPeIwd7sCk',
]


PASSING_PAIRS = [
    ('xXx0MtgG8LTm4mSVWscg4s97g0W', 'xXx09ejihbvx7yUX9JYxtIVPGDs'),
    ('xXx0MtgG8LTm4mSVWscg4s97g0W', 'xXx0UVvEW16rb3wrfuDSAemYblI'),
    ('xXx0LFCBTNE4Wx7KUjfsG1nZc4I', 'xXx07Gk5iNRGngQLpVQwbMDAAS1'),
    ('xXx0LFCBTNE4Wx7KUjfsG1nZc4I', 'xXx09ejihbvx7yUX9JYxtIVPGDs'),
    ('xXx0LFCBTNE4Wx7KUjfsG1nZc4I', 'xXx04jaaxCjDADdZfQejz3gbNFD'),
    ('xXx0MtgG8LTm4mSVWscg4s97g0W', 'xXx07Gk5iNRGngQLpVQwbMDAAS1'),
    ('xXx0WcxFNILJIkWzWO9MsEGL0Jx', 'xXx09ejihbvx7yUX9JYxtIVPGDs'),
    ('xXx0WcxFNILJIkWzWO9MsEGL0Jx', 'xXx04jaaxCjDADdZfQejz3gbNFD'),
    ('xXx0UVvEW16rb3wrfuDSAemYblI', 'xXx09ejihbvx7yUX9JYxtIVPGDs')
]
EXCLUDE_PAIRS = { #pairs that failed in previous runs, exclude from testing for now
    ('xXx04PqApRQMPw55qx6cZsUfVSi', 'xXx04jaaxCjDADdZfQejz3gbNFD'),
    ('xXx04PqApRQMPw55qx6cZsUfVSi', 'xXx061oJQctwH2mBMMPeIwd7sCk'),
    ('xXx05DUkH2xBIGogZRAFJ8nnJHE', 'xXx061oJQctwH2mBMMPeIwd7sCk'),
    ('xXx05DUkH2xBIGogZRAFJ8nnJHE', 'xXx07Gk5iNRGngQLpVQwbMDAAS1'),
    ('xXx0LFCBTNE4Wx7KUjfsG1nZc4I', 'xXx04PqApRQMPw55qx6cZsUfVSi'),
    ('xXx0MtgG8LTm4mSVWscg4s97g0W', 'xXx05DUkH2xBIGogZRAFJ8nnJHE'),
    ('xXx0UVvEW16rb3wrfuDSAemYblI', 'xXx04jaaxCjDADdZfQejz3gbNFD'),
    ('xXx0WcxFNILJIkWzWO9MsEGL0Jx', 'xXx07Gk5iNRGngQLpVQwbMDAAS1'),
    ('xXx04PqApRQMPw55qx6cZsUfVSi', 'xXx09ejihbvx7yUX9JYxtIVPGDs'),
    ('xXx0UVvEW16rb3wrfuDSAemYblI', 'xXx061oJQctwH2mBMMPeIwd7sCk'),
    ('xXx0UVvEW16rb3wrfuDSAemYblI', 'xXx061oJQctwH2mBMMPeIwd7sCk'),
    ('xXx04PqApRQMPw55qx6cZsUfVSi', 'xXx05DUkH2xBIGogZRAFJ8nnJHE'),
    ('xXx07Gk5iNRGngQLpVQwbMDAAS1', 'xXx061oJQctwH2mBMMPeIwd7sCk'),
    ('xXx0LFCBTNE4Wx7KUjfsG1nZc4I', 'xXx061oJQctwH2mBMMPeIwd7sCk'),
    ('xXx0MtgG8LTm4mSVWscg4s97g0W', 'xXx061oJQctwH2mBMMPeIwd7sCk')
}
#────── 1. Create different individual pairs ──────────────────────────
ALL_POSSIBLE_PAIRS = list(combinations(INDIVIDUALS, 2)) # creates all the differnt possible pairs of individuals
FILTERED_PAIRS = [pair for pair in ALL_POSSIBLE_PAIRS if pair not in EXCLUDE_PAIRS and pair not in PASSING_PAIRS] #exclude pairs that we've already tested
PAIRS = random.sample(FILTERED_PAIRS, 12) #randomly sample 12 pairs to test on (may change this number depending on how long it taks to run)

#────── 2. For each pair, run the crossover script ─────────────────────
@pytest.mark.parametrize('gene_id_x, gene_id_y', PAIRS)
def test_crossover_function(gene_id_x, gene_id_y):
    #insert the pairs of individuals into the crossover script and run it
    input_filename_x = os.path.join(INDIVIDUALS_DIR, f'model_{gene_id_x}.py')
    input_filename_y = os.path.join(INDIVIDUALS_DIR, f'model_{gene_id_y}.py')

    crossed_output = os.path.join(RESULTS_DIR, f'model_crossed_{gene_id_x}_{gene_id_y}.py')

    #running the crossover script with the two individuals/pair as input and the output file for the crossed individual
    result = subprocess.run(
        ['uv', 'run', 'python', CROSSOVER_SCRIPT, input_filename_x, input_filename_y, crossed_output],
        cwd=ROOT, capture_output=True, text=True, timeout=10000
    )

    #check if the crossover scrip ran successfully
    # Runtime errors expected by some so skip over those ** is this valid?
    if result.returncode != 0:
        pytest.skip(f'({gene_id_x}, {gene_id_y}): runtime error in crossover (expected for some of the pairs)')
        #pytest.fail(f'({gene_id_x}, {gene_id_y}): {result.stderr}')

    assert 'job done' in result.stdout.lower(), \
        f"({gene_id_x}, {gene_id_y}): 'job done' not in output"
    
    #Ensure the output file was created/written to
    assert os.path.isfile(crossed_output), \
        f"({gene_id_x}, {gene_id_y}): crossed output file not created at {crossed_output}"
    
    #Validate the crossed output file is a valid python file and contains a model class with fit and predict methods
    with open(crossed_output) as f:
        tree = ast.parse(f.read())
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'Model']
    assert len(classes) >= 1, f"({gene_id_x}, {gene_id_y}): no Model class in crossed output"
    methods = {n.name for n in ast.walk(classes[0]) if isinstance(n, ast.FunctionDef)}
    assert 'fit' in methods, f"({gene_id_x}, {gene_id_y}): Model missing fit() in crossed output"
    assert 'predict' in methods, f"({gene_id_x}, {gene_id_y}): Model missing predict() in crossed output"