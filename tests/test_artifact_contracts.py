"""Regression checks for generated model artifacts and notebook boundaries."""
import json
from pathlib import Path

import nbformat
import pytest

from src.classification import load_classifier_artifact


@pytest.mark.parametrize("notebook_path", sorted(Path("notebooks").glob("*.ipynb")))
def test_notebooks_do_not_write_model_artifacts(notebook_path):
    notebook = nbformat.read(notebook_path, as_version=4)
    forbidden = ("save_classifier_artifact", "save_forecasting_artifact", "joblib.dump")
    source = "\n".join(cell.source for cell in notebook.cells)
    assert not [token for token in forbidden if token in source], notebook_path


def test_deployed_classifier_artifact_matches_report():
    artifact_path = Path("models/no_show_model.joblib")
    if not artifact_path.exists():
        pytest.skip("Model artifact not yet trained. Run run_pipeline.py first.")

    artifact = load_classifier_artifact(str(artifact_path))
    metadata = artifact["metadata"]
    with Path("outputs/classification_report.json").open(encoding="utf-8") as handle:
        report = json.load(handle)

    deployed_variant = report["deployed_variant"]
    assert deployed_variant == "no_place"
    assert metadata["variant"] == "no_place"
    assert "place_clean" not in list(artifact["pipeline"].named_steps["preprocessor"].feature_names_in_)
    expected = report["variants"][deployed_variant]["models"][metadata["model_name"]]
    assert metadata["metrics"] == expected
