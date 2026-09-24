"""The interface's own logic: the workflow stepper, run stages, dataset status, help
terms, and the recount of the specification space shown on the configure page.

None of this computes a result, but several pieces restate one — the recount above
all — so they are checked against the engine rather than trusted.
"""
from types import SimpleNamespace

import pytest

from app import services, ui
from app.core.grid import capabilities_for, enumerate_grid
from app.core.preprocess import MatrixBuilder


# --- the specification space ---------------------------------------------------
@pytest.mark.parametrize("name", list(services.DEMO_DATASETS))
def test_the_configure_page_recount_matches_the_engine(name):
    """The fork cards multiply out to exactly what enumerate_grid enumerates."""
    dataset = services.load_demo(name)
    builder = MatrixBuilder(dataset)
    capabilities = capabilities_for(dataset)
    previews = {
        mode: enumerate_grid(builder, mode=mode, capabilities=capabilities,
                             covariate_columns=dataset.covariate_columns)[1]
        for mode in ("quick", "full", "covariate")
    }
    space = ui.specification_space(builder, capabilities, dataset.covariate_columns, previews)
    for mode, entry in space.items():
        report = previews[mode]
        assert entry["total"] == report.n_enumerated, (name, mode)
        assert entry["matches"], (name, mode)
        # Seven cards, numbered as the specification numbers the forks.
        assert [card["number"] for card in entry["cards"]] == [1, 2, 3, 4, 5, 6, 7]
        # The "x N corrections" the page prints must be exact, or not printed at all.
        if entry["per_fit"]:
            assert report.n_fits * entry["per_fit"] == report.n_valid


def test_covariate_mode_counts_scale_with_the_subsets():
    """The page recomputes the count as covariates are unticked: valid per subset
    times the number of subsets. That only holds if pruning ignores covariates."""
    dataset = services.load_demo("t2d_covariates")
    builder = MatrixBuilder(dataset)
    capabilities = capabilities_for(dataset)
    columns = list(dataset.covariate_columns)
    _, everything = enumerate_grid(builder, mode="covariate", capabilities=capabilities,
                                   covariate_columns=columns)
    _, fewer = enumerate_grid(builder, mode="covariate", capabilities=capabilities,
                              covariate_columns=columns[:2])
    per_subset = everything.n_valid // 2 ** len(columns)
    assert everything.n_valid == per_subset * 2 ** len(columns)
    assert fewer.n_valid == per_subset * 2 ** 2


# --- the stepper ---------------------------------------------------------------
def _by_key(steps):
    return {step["key"]: step for step in steps}


def test_the_current_step_is_never_a_link():
    for current in ("validate", "configure", "run", "results"):
        steps = _by_key(ui.workflow_steps(current, "tok", "done"))
        assert steps[current]["state"] == "current"
        assert steps[current]["href"] == ""


def test_configure_is_not_offered_while_a_run_is_in_progress():
    """Submitting the form again would start a second run over the first."""
    steps = _by_key(ui.workflow_steps("run", "tok", "running"))
    assert steps["configure"]["href"] == ""
    assert steps["validate"]["href"] == "/validate/tok"
    assert steps["results"]["href"] == ""


def test_results_are_only_linked_once_they_exist():
    assert _by_key(ui.workflow_steps("configure", "tok", "uploaded"))["results"]["href"] == ""
    done = _by_key(ui.workflow_steps("configure", "tok", "done"))
    assert done["results"]["href"] == "/results/tok"
    assert done["results"]["state"] == "done"


# --- run stages ------------------------------------------------------------------
def _job(status, message, progress):
    return SimpleNamespace(status=status, message=message, progress=progress)


def test_the_fitting_stage_reports_the_engines_own_counter():
    stages = ui.run_stages(_job("running", "Fitting specifications: matrix 12 of 160", 0.4))
    states = {stage["key"]: stage for stage in stages}
    assert states["queue"]["state"] == "done"
    assert states["grid"]["state"] == "done"
    assert states["fit"]["state"] == "current"
    assert states["fit"]["detail"] == "matrix 12 of 160"
    assert states["tiers"]["state"] == "pending"


@pytest.mark.parametrize("message,key", [
    ("Queued — waiting for a free analysis slot", "queue"),
    ("Starting", "grid"),
    ("Enumerating the specification grid", "grid"),
    ("Assembling results", "assemble"),
    ("Computing robustness", "tiers"),
    ("Computing robustness tiers", "tiers"),
    ("Attributing variance to the forks", "attribution"),
    ("Saving results", "save"),
])
def test_every_message_the_engine_emits_has_a_stage(message, key):
    current = [s for s in ui.run_stages(_job("running", message, 0.5)) if s["state"] == "current"]
    assert [s["key"] for s in current] == [key]


def test_a_finished_run_has_every_stage_done_and_a_failed_one_says_where():
    assert {s["state"] for s in ui.run_stages(_job("done", "Complete", 1.0))} == {"done"}
    failed = ui.run_stages(_job("error", "Attributing variance to the forks", 0.97))
    assert [s["key"] for s in failed if s["state"] == "error"] == ["attribution"]


# --- dataset status ----------------------------------------------------------------
@pytest.mark.parametrize("worst,key,tone", [
    ("ok", "ready", "green"), ("note", "ready", "green"),
    ("caution", "review", "amber"), ("serious", "review", "red"),
])
def test_dataset_status_follows_the_worst_readiness_level(worst, key, tone):
    status = ui.dataset_status(SimpleNamespace(worst=worst))
    assert (status["key"], status["tone"]) == (key, tone)
    # Readiness never blocks a run, and the status must not claim that it does.
    assert "blocks" not in status["sentence"] or "Nothing here blocks" in status["sentence"]


# --- help terms ----------------------------------------------------------------------
def test_every_help_term_the_interface_uses_is_defined():
    for key in ("p_value", "q_value", "clr", "tmm", "taxonomic_rank", "robustness",
                "specification", "fdr", "prevalence_filter", "rarefaction", "pruned",
                "direction_agreement", "effect_size"):
        entry = ui.help_term(key)
        assert entry["short"], key


def test_the_adjusted_p_value_is_not_passed_off_as_storeys_q():
    assert "does not compute Storey" in ui.help_term("q_value")["long"]


def test_the_glossary_gains_the_interface_terms():
    from app.core.glossary import glossary_sections

    headings = [s["heading"] for s in ui.glossary_with_ui_terms(glossary_sections())]
    assert headings[-1] == ui.UI_GLOSSARY_HEADING
    assert len(headings) == len(glossary_sections()) + 1
