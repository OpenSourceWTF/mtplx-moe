from __future__ import annotations

from datetime import date

import pytest

from mtplx.experiments.schema import ExperimentStatus, load_experiment


def _recipe(tmp_path, text: str):
    path = tmp_path / "recipe.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_complete_active_recipe(tmp_path):
    recipe = load_experiment(
        _recipe(
            tmp_path,
            '''
[experiment]
id = "compiled-verify-control"
title = "Compiled verify disabled control"
status = "active"
owner = "runtime"
tracking = "https://github.com/davidtai/MTPLX/issues/90"
created = "2026-07-16"
review_after = "2026-08-16"
models = ["qwen3-next"]
purpose = "Isolate compiled verify."
[settings]
"verify.compiled.mode" = "off"
''',
        ),
        today=date(2026, 7, 16),
    )
    assert recipe.status is ExperimentStatus.ACTIVE
    assert recipe.settings == {"verify.compiled.mode": "off"}


def test_rejects_executable_content(tmp_path):
    path = _recipe(
        tmp_path,
        '''
[experiment]
id = "unsafe"
title = "Unsafe"
status = "active"
owner = "runtime"
tracking = "https://github.com/davidtai/MTPLX/issues/90"
created = "2026-07-16"
review_after = "2026-08-16"
models = ["qwen3-next"]
purpose = "Unsafe fixture."
[settings]
"runtime.profile" = "sustained"
[shell]
command = "echo unsafe"
''',
    )
    with pytest.raises(ValueError, match="unsupported top-level table: shell"):
        load_experiment(path, today=date(2026, 7, 16))


def test_active_recipe_must_have_future_review_date(tmp_path):
    path = _recipe(
        tmp_path,
        '''
[experiment]
id = "expired"
title = "Expired"
status = "active"
owner = "runtime"
tracking = "https://github.com/davidtai/MTPLX/issues/90"
created = "2026-01-01"
review_after = "2026-01-02"
models = ["qwen3-next"]
purpose = "Expired fixture."
[settings]
"runtime.profile" = "sustained"
''',
    )
    with pytest.raises(ValueError, match="review date has passed"):
        load_experiment(path, today=date(2026, 7, 16))


def test_rejects_nested_or_array_setting_values(tmp_path):
    path = _recipe(
        tmp_path,
        '''
[experiment]
id = "nested"
title = "Nested"
status = "retained"
owner = "runtime"
tracking = "https://github.com/davidtai/MTPLX/issues/90"
created = "2026-07-16"
review_after = "2026-08-16"
models = ["qwen3-next"]
purpose = "Nested fixture."
[settings]
"runtime.profile" = ["sustained"]
''',
    )
    with pytest.raises(ValueError, match="scalar values"):
        load_experiment(path, today=date(2026, 7, 16))
