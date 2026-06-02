import sys
# We are two directories down from run_improved, but running from its location
sys.path.append('./')
print(sys.path)
import run_improved
import os

def test_individual(monkeypatch, tmp_path):
    monkeypatch.setattr(run_improved, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(run_improved, "wait_for_llm_server", lambda: True)
    monkeypatch.setattr(
        run_improved.subprocess,
        "run",
        lambda *args, **kwargs: run_improved.subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="Submitted batch job 12345",
            stderr="",
        ),
    )

    individual = run_improved.toolbox.individual(llm_model=run_improved.DEFAULT_LLM_MODEL)
    model_path = os.path.join(run_improved.VARIANT_DIR, f'{run_improved.MODEL}_{individual[0]}.py')
    script_path = os.path.join(run_improved.OUTPUT_DIR, str(run_improved.GENERATION), f'{individual[0]}.sh')
    assert os.path.exists(script_path)
    assert model_path in open(script_path).read()
