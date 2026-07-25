import json
import subprocess
import sys
import zipfile


def test_built_wheel_contains_expert_profiles(tmp_path):
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
        check=True,
    )
    wheel = next(tmp_path.glob("mtplx-2.3.1rc1-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        payload = json.loads(
            archive.read("mtplx/data/expert_profiles.json")
        )
    assert [row["name"] for row in payload["profiles"]] == [
        "hy3-oq2e-64",
        "hy3-oq2e-88",
        "hy3-oq2e-96",
    ]
