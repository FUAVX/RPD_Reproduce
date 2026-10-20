"""Execute notebook preparation against local Git + empty dataset layout.

No pip installs, GPU calls, dataset reading or training. Training subprocesses
are mocked only when checking command construction and artifact packaging.
Run from repo root with PyYAML; nbformat validation is optional when installed.
"""
import ast
import contextlib
import copy
import io
import json
import re
import subprocess
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = []
    pins = set()
    with tempfile.TemporaryDirectory(prefix="p3-notebook-") as temporary:
        fixture = Path(temporary)
        dataset = fixture / "input" / "phenobench"
        for relative in ("train/images", "train/semantics", "val/images", "val/semantics", "test/images"):
            (dataset / relative).mkdir(parents=True)
        notebooks = sorted((ROOT / "notebook").glob("kaggle_train_p3_*.ipynb"))
        assert len(notebooks) == 2
        for notebook in notebooks:
            nb = json.loads(notebook.read_text())
            for i, cell in enumerate(nb["cells"]):
                if cell["cell_type"] == "code":
                    assert cell["execution_count"] is None and not cell["outputs"]
                    compile("".join(cell["source"]), f"{notebook.name}:{i}", "exec")
            try:
                import nbformat
            except ImportError:
                pass
            else:
                nbformat.validate(nbformat.from_dict(nb))
            setup = "".join(nb["cells"][4]["source"])
            pin = re.search(r'SOURCE_COMMIT = "([0-9a-f]{40})"', setup).group(1)
            pins.add(pin)
            assert 'GITHUB_REPO = "https://github.com/FUAVX/RPD_Reproduce.git"' in setup
            for obsolete in ("SOURCE_INPUT", "p3_package_manifest", "unpack_source"):
                assert obsolete not in setup
            # Execute the existing dependency cell's imports/helper, omit pip call.
            helper = ast.parse("".join(nb["cells"][2]["source"]))
            assert isinstance(helper.body[-1], ast.Expr)
            helper.body.pop()
            namespace = {}
            exec(compile(helper, "notebook_run_helper", "exec"), namespace)
            local_setup = setup.replace('GITHUB_REPO = "https://github.com/FUAVX/RPD_Reproduce.git"',
                                        f'GITHUB_REPO = {str(ROOT)!r}')
            local_setup = local_setup.replace('"/kaggle/working"', repr(str(fixture / "working")))
            local_setup = local_setup.replace('"/kaggle/input"', repr(str(fixture / "input")))
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(local_setup, notebook.name, "exec"), namespace)
            run_dir, repo = namespace["RUN_DIR"], namespace["REPO_DIR"]
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
            assert manifest["source_commit"] == pin
            assert manifest["source_diff"] == manifest["source_status"] == ""
            assert manifest["resume_checkpoint"] is None
            assert manifest["effective_batch"] == 8
            assert not (repo / "results").exists()
            assert namespace["source_state"]()[0] == pin
            active = yaml.safe_load((run_dir / "active_config.yaml").read_text())
            original = yaml.safe_load((repo / "configs/p3_scenarios" / f'{manifest["scenario"]}.yaml').read_text())
            expected = copy.deepcopy(original)
            expected["data"].update(path_to_dataset=str(dataset), num_workers=2)
            expected["train"]["devices"] = 2
            expected["val"]["devices"] = 1
            assert active == expected
            # Dirty tracked code must be rejected before preflight/train.
            factory = repo / "models/__init__.py"
            original_factory = factory.read_bytes()
            factory.write_bytes(original_factory + b"\n# dirty test\n")
            try:
                namespace["source_state"]()
                raise AssertionError("Dirty source accepted")
            except RuntimeError as error:
                assert "remain clean" in str(error)
            factory.write_bytes(original_factory)
            # Resume validation runs from the actual cell without cloning again.
            resume_start = setup.index("resume_values =")
            resume_end = setup.index("MANIFEST = dict")
            resume_check = setup[resume_start:resume_end]
            checkpoint = run_dir / "fixture.ckpt"
            checkpoint.write_bytes(b"fixture only, never loaded")
            namespace.update(RESUME_CHECKPOINT=str(checkpoint),
                             RESUME_CONFIG=str(run_dir / "active_config.yaml"),
                             RESUME_MANIFEST=str(run_dir / "run_manifest.json"))
            exec(resume_check, namespace)  # same source/config is accepted
            namespace["RESUME_MANIFEST"] = ""
            try:
                exec(resume_check, namespace)
                raise AssertionError("Incomplete resume accepted")
            except ValueError as error:
                assert "requires checkpoint" in str(error)
            namespace["RESUME_MANIFEST"] = str(run_dir / "run_manifest.json")
            wrong = dict(manifest, source_commit="0" * 40)
            (run_dir / "run_manifest.json").write_text(json.dumps(wrong))
            try:
                exec(resume_check, namespace)
                raise AssertionError("Wrong source resume accepted")
            except ValueError as error:
                assert "source/scenario mismatch" in str(error)
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
            different_cfg = copy.deepcopy(active)
            different_cfg["backbone"]["name"] = "OtherScenario"
            wrong_cfg = run_dir / "wrong_config.yaml"
            wrong_cfg.write_text(yaml.safe_dump(different_cfg))
            namespace["RESUME_CONFIG"] = str(wrong_cfg)
            try:
                exec(resume_check, namespace)
                raise AssertionError("Wrong scenario resume accepted")
            except ValueError as error:
                assert "config mismatch" in str(error)
            # Train is blocked without completed GPU preflight, even if a stale
            # report file exists. No GPU is claimed or executed by these checks.
            namespace.update(RESUME_CHECKPOINT="", RESUME_CONFIG="", RESUME_MANIFEST="")
            train = "".join(nb["cells"][8]["source"])
            (run_dir / "gpu_preflight.json").write_text('{}')
            try:
                exec(train, namespace)
                raise AssertionError("Missing preflight accepted")
            except RuntimeError as error:
                assert "preflight cell successfully" in str(error)
            # Mock process dispatch to inspect fresh/resume CLI and ZIP contents.
            commands = []
            def mock_run(command, **kwargs):
                commands.append(command)
            namespace["run"] = mock_run
            namespace["MANIFEST"]["preflight_passed"] = True
            with contextlib.redirect_stdout(io.StringIO()):
                exec(train, namespace)
            assert "--resume" not in commands[-1] and "--ckpt_path" not in commands[-1]
            assert commands[-1][-2:] == ["--devices", "2"]
            namespace["RESUME_CHECKPOINT"] = str(checkpoint)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(train, namespace)
            assert commands[-1][-3:] == ["--ckpt_path", str(checkpoint), "--resume"]
            with namespace["zipfile"].ZipFile(namespace["artifact_path"]) as archive:
                names = archive.namelist()
                assert "run_manifest.json" in names and "active_config.yaml" in names
                assert not any(p.startswith("source/") for p in names)
            reports.append(dict(notebook=notebook.name, source_commit=pin,
                                checks="syntax, local Git clone/pin, recipe, provenance, dirty-source rejection, resume guards, preflight gate, mocked CLI/artifacts",
                                status="passed"))
    assert len(pins) == 1, "Both experiments must use the same reviewed source"
    result = dict(checks=reports, fixture="empty dataset directory layout; local Git remote",
                  pip_gpu_dataset_training_executed=False, training_command_mocked=True)
    output = ROOT / "docs/p3/notebook_checks.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Both notebook preparation checks passed: {output}")


if __name__ == "__main__":
    main()
