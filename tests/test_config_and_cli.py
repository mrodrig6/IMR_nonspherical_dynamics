"""Configuration loading, the bundled cases, and the command-line interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imrns.cli import main
from imrns.config import CaseConfig, Material, load_case
from imrns.runner import run_case
from imrns.scales import compute_scales, rescale_radial_history

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CASES = sorted(EXAMPLES.glob("*.toml"))


def test_examples_exist():
    assert len(CASES) == 18, f"expected 18 bundled cases, found {len(CASES)}"


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.stem)
def test_every_bundled_case_loads_and_has_sane_groups(path):
    config = load_case(path)
    assert config.name == path.stem
    assert config.description
    assert len(config.ep0) == len(config.modes) == len(config.epd0)
    s = config.scales()
    assert s.We > 0 and np.isfinite(s.We)
    # At most one of Ca and Re may be infinite: the rotational closure needs
    # some deviatoric stress to be well posed.
    assert np.isfinite(s.Ca) or np.isfinite(s.Re)


def test_unknown_keys_are_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('modes = [5]\n[material]\nviscosity = 0.1\n')
    with pytest.raises(ValueError, match="unknown Material keys"):
        load_case(bad)


def test_unknown_top_level_keys_are_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('modes = [5]\ntypoed_key = 3\n')
    with pytest.raises(ValueError, match="unknown top-level keys"):
        load_case(bad)


def test_scalar_initial_amplitude_is_broadcast(tmp_path):
    case = tmp_path / "c.toml"
    case.write_text('modes = [2, 5, 8]\nep0 = [0.01]\n')
    assert load_case(case).ep0 == [0.01, 0.01, 0.01]


def test_imrv2_cases_ask_for_a_history():
    config = load_case(EXAMPLES / "lic_viscoelastic_single.toml")
    with pytest.raises(ValueError, match="needs a radial history from IMRv2"):
        config.radial_history()


def test_run_case_produces_both_closures():
    config = CaseConfig(modes=[5], ep0=[0.01])
    config.numerics.xN = 24
    config.drive.n_steps = 60
    config.drive.t_end = 1.0
    result = run_case(config)
    assert result.rotational.ep.shape == (1, 60)
    assert result.irrotational is not None
    assert result.irrotational.ep.shape == (1, 60)


def test_case_result_round_trips_through_npz(tmp_path):
    config = CaseConfig(modes=[5], ep0=[0.01])
    config.numerics.xN = 16
    config.drive.n_steps = 40
    config.drive.t_end = 0.5
    result = run_case(config)
    path = result.save(tmp_path / "out.npz")
    data = np.load(path)
    np.testing.assert_allclose(data["ep"], result.rotational.ep)
    assert "ep_irrotational" in data


def test_rescaling_a_radial_history_is_self_inverse():
    t = np.linspace(0, 1, 11)
    R, Rd, Rdd = 1 + 0.1 * t, 0.1 * np.ones_like(t), np.zeros_like(t)
    a = rescale_radial_history(t, R, Rd, Rdd, R_from=150e-6, R_to=50e-6)
    b = rescale_radial_history(*a, R_from=50e-6, R_to=150e-6)
    for original, restored in zip((t, R, Rd, Rdd), b):
        np.testing.assert_allclose(restored, original, rtol=1e-14)


def test_pressure_scale_equals_ambient_pressure():
    """pc = rho (Lc/tc)^2 reduces to p_inf by construction."""
    s = compute_scales(Req=50e-6, rho=1048, p_inf=101325, mu=0.015, G=2e3, sigma=0.056)
    assert s.pc == pytest.approx(101325.0)


def test_vanishing_properties_give_infinite_groups():
    s = Material(mu=0.0, G=0.0).scales(50e-6)
    assert np.isinf(s.Ca) and np.isinf(s.Re) and np.isfinite(s.We)


def test_cli_list_and_info(capsys):
    assert main(["list"]) == 0
    assert "free_viscoelastic_single" in capsys.readouterr().out
    assert main(["info", str(EXAMPLES / "free_viscous_multi.toml")]) == 0
    assert "groups" in capsys.readouterr().out


def test_cli_run(capsys, tmp_path):
    code = main(["run", str(EXAMPLES / "free_viscoelastic_single.toml"),
                 "--xN", "16", "--steps", "40", "--no-compare",
                 "-o", str(tmp_path / "r.npz")])
    assert code == 0
    assert "completed" in capsys.readouterr().out
    assert (tmp_path / "r.npz").exists()


def test_cli_reports_configuration_errors_without_a_traceback(capsys):
    assert main(["run", str(EXAMPLES / "lic_viscous_single.toml")]) == 1
    assert "needs a radial history from IMRv2" in capsys.readouterr().err
