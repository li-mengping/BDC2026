from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = spec_from_file_location(name, path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_consumes_every_approved_scenario() -> None:
    parser = _load_module('acceptance_parser', ROOT / 'acceptance-pipeline' / 'parse_specs.py')
    generator = _load_module('acceptance_generator', ROOT / 'acceptance-pipeline' / 'generate_tests.py')
    cases = parser.parse_specs(ROOT / 'specs')

    rendered = generator.render_tests(cases)

    assert {case['id'] for case in cases} == set(generator.SCENARIO_RENDERERS)
    for case in cases:
        assert f"source: {case['source']}:{case['line']}" in rendered


def test_specs_use_unique_domain_ids() -> None:
    parser = _load_module('acceptance_parser_unique', ROOT / 'acceptance-pipeline' / 'parse_specs.py')
    cases = parser.parse_specs(ROOT / 'specs')

    ids = [case['id'] for case in cases]

    assert ids
    assert len(ids) == len(set(ids))
