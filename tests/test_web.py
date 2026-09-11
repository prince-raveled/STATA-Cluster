"""Web layer: routes, API, downloads, and the §8 / §18 guarantees they must keep."""
from __future__ import annotations

import io
import json
import zipfile

import pytest

pytest.importorskip("httpx", reason="TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app import config, db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("microverse-data")
    config.DATA_DIR = root
    config.JOBS_DIR = root / "jobs"
    config.DATABASE_URL = f"sqlite:///{(root / 'jobs.sqlite').as_posix()}"
    db.init()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def finished(client):
    """A completed Quick-mode run on the smallest demo dataset."""
    response = client.get("/demo/ibd_genus", follow_redirects=False)
    token = response.headers["location"].rsplit("/", 1)[-1]
    client.post(f"/run/{token}", data={"mode": "quick"}, follow_redirects=False)
    job = db.get_job(token)
    assert job.status == "done", job.error
    return token


# --- pages ---------------------------------------------------------------
def test_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "multiverse" in response.text.lower()


def test_about_page_states_the_norm(client):
    response = client.get("/about")
    assert response.status_code == 200
    assert "Report the distribution, not your favourite point in it" in response.text


def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"


def test_openapi_is_served(client):
    schema = client.get("/api/openapi.json").json()
    assert schema["info"]["title"] == "MicroVerse"
    assert "/api/jobs" in schema["paths"]


def test_configure_page_reports_both_grid_counts(client):
    response = client.get("/demo/ibd_genus", follow_redirects=True)
    assert response.status_code == 200
    assert "enumerated" in response.text
    assert "statistically valid" in response.text


def test_results_page_renders(client, finished):
    response = client.get(f"/results/{finished}")
    assert response.status_code == 200
    # The sections a results page must always carry. Headings are plain language now,
    # so these match what a researcher actually sees rather than specification numbers.
    for fragment in ("how each taxon held up", "which choice is driving this",
                     "one taxon, every analysis", "what was actually run",
                     "methods paragraph", "every taxon, sortable"):
        assert fragment in response.text.lower(), fragment
    # The orientation panel and the inline definitions are part of the contract: a
    # first-time reader must be able to find out what a "specification" is.
    assert "WHAT HAPPENED" in response.text
    assert 'class="dfn"' in response.text
    assert 'id="curve"' in response.text
    assert 'id="taxa-table"' in response.text


def test_curve_endpoint_shape(client, finished):
    payload = client.get(f"/results/{finished}/curve/0").json()
    assert payload["n_specs"] > 0
    assert len(payload["effect"]) == len(payload["significant"]) == payload["n_plotted"]
    assert set(payload["forks"]) == set(payload["fork_labels"])
    for fork, values in payload["forks"].items():
        assert len(values) == payload["n_plotted"]
        assert set(values) <= set(payload["categories"][fork])


def test_curve_effects_are_sorted(client, finished):
    effects = client.get(f"/results/{finished}/curve/0").json()["effect"]
    assert effects == sorted(effects)


# --- §8: refusals are sentences, not stack traces -------------------------
def test_bad_abundance_file_is_refused_with_a_message(client):
    response = client.post(
        "/upload",
        files={
            "abundance": ("junk.tsv", b"not a table at all", "text/tab-separated-values"),
            "metadata": ("m.tsv", b"sample_id\tgroup\nS1\ta\nS2\tb\n",
                         "text/tab-separated-values"),
        },
    )
    assert response.status_code == 422
    assert "Traceback" not in response.text


def test_too_few_samples_is_refused_with_the_reason(client):
    abundance = "taxon\t" + "\t".join(f"S{i}" for i in range(8)) + "\n"
    for t in range(12):
        abundance += f"T{t}\t" + "\t".join(str(10 + t + i) for i in range(8)) + "\n"
    metadata = "sample_id\tgroup\n" + "".join(
        f"S{i}\t{'a' if i < 4 else 'b'}\n" for i in range(8)
    )
    response = client.post(
        "/upload",
        files={"abundance": ("a.tsv", abundance.encode(), "text/tab-separated-values"),
               "metadata": ("m.tsv", metadata.encode(), "text/tab-separated-values")},
    )
    assert response.status_code == 422
    assert "at least 10" in response.text
    assert "Traceback" not in response.text


def test_api_errors_are_json(client):
    response = client.post(
        "/api/jobs",
        files={"abundance": ("a.tsv", b"nope", "text/plain"),
               "metadata": ("m.tsv", b"sample_id\tgroup\nS1\ta\n", "text/plain")},
        data={"mode": "quick"},
    )
    assert response.status_code == 422
    assert "error" in response.json()


def test_unknown_job_is_a_404(client):
    assert client.get("/results/deadbeefdeadbeefdeadbeef").status_code == 404
    assert client.get("/job/deadbeefdeadbeefdeadbeef").status_code == 404
    assert client.get("/api/jobs/deadbeefdeadbeefdeadbeef").status_code == 404


def test_a_malformed_token_never_reaches_the_filesystem(client):
    """Tokens name a directory, so anything that is not 24 hex characters is refused
    before any path is built from it."""
    for bad in ("../../etc/passwd", "..", ".", "not-a-token", "a" * 200, "",
                "ABCDEF0123456789abcdef01", "0123456789abcdef0123456"):
        assert not db.valid_token(bad), bad
        assert db.get_job(bad) is None
    assert db.valid_token("0123456789abcdef01234567")

    for token in ("not-a-token", "a" * 60, "0123456789ABCDEF01234567"):
        assert client.get(f"/results/{token}").status_code == 404
        assert client.get(f"/job/{token}").status_code == 404


# --- API -----------------------------------------------------------------
def test_api_info_declares_the_policy(client):
    info = client.get("/api/info").json()
    assert info["policy"]["best_specification_export"].startswith("never")
    assert info["policy"]["two_group_only"] is True
    assert set(info["modes"]) == {"quick", "full", "covariate"}
    assert len(info["forks"]) == 8  # seven forks, with rarefaction seeds listed


def test_api_results_carry_the_manifest(client, finished):
    payload = client.get(f"/api/jobs/{finished}/results").json()
    assert payload["verdict"]
    grid = payload["manifest"]["grid"]
    assert grid["valid"] + grid["pruned"] == grid["enumerated"]
    assert payload["manifest"]["tiers"]
    assert payload["n_taxa_total"] >= payload["n_taxa_returned"] > 0
    assert payload["taxa"]


def test_api_results_can_filter_by_tier(client, finished):
    payload = client.get(f"/api/jobs/{finished}/results", params={"tier": "robust"}).json()
    assert all(row["robustness_tier"] == "ROBUST" for row in payload["taxa"])


def test_api_curve_rejects_an_unknown_taxon(client, finished):
    assert client.get(f"/api/jobs/{finished}/curve/999999").status_code == 404


# --- §16.5 downloads, and §18 -------------------------------------------
@pytest.mark.parametrize("kind", ["robustness", "specifications", "long", "attribution",
                                  "methods", "bundle"])
def test_downloads(client, finished, kind):
    response = client.get(f"/download/{finished}/{kind}")
    assert response.status_code == 200
    assert len(response.content) > 100
    assert "attachment" in response.headers["content-disposition"]


def test_bundle_contains_the_whole_distribution(client, finished):
    archive = zipfile.ZipFile(io.BytesIO(client.get(f"/download/{finished}/bundle").content))
    names = set(archive.namelist())
    assert names == {"README.txt", "methods.txt", "taxa_robustness.csv",
                     "specifications.csv", "results_long.csv.gz", "attribution.csv",
                     "manifest.json"}
    manifest = json.loads(archive.read("manifest.json"))
    assert manifest["grid"]["valid"] > 0
    readme = archive.read("README.txt").decode()
    assert "does NOT contain" in readme and "best" in readme


def test_there_is_no_best_specification_download(client, finished):
    for kind in ("best", "best_spec", "optimal", "recommended"):
        response = client.get(f"/download/{finished}/{kind}")
        assert response.status_code == 404
        # The apostrophes are HTML-escaped in the rendered error page.
        assert "best specification" in response.text
        assert "by design" in response.text


def test_methods_paragraph_describes_the_multiverse(client, finished):
    text = client.get(f"/download/{finished}/methods").text
    assert "multiverse" in text.lower()
    assert "distribution of results" in text
    assert "not a preferred point" in text


def test_specifications_export_covers_every_specification(client, finished):
    import pandas as pd

    frame = pd.read_csv(io.BytesIO(client.get(f"/download/{finished}/specifications").content))
    manifest = client.get(f"/api/jobs/{finished}/results").json()["manifest"]
    assert len(frame) == manifest["grid"]["valid"]
    expected = {"rarefaction", "transform", "method", "fdr_method", "n_significant"}
    assert expected <= set(frame.columns)


# --- job lifecycle -------------------------------------------------------
def test_job_progress_fragment(client, finished):
    response = client.get(f"/job/{finished}/progress")
    assert response.status_code == 200
    assert "results" in response.text


def test_finished_job_page_redirects_to_results(client, finished):
    response = client.get(f"/job/{finished}", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].endswith(f"/results/{finished}")


def test_declared_pipeline_is_located(client):
    response = client.get("/demo/ibd_genus", follow_redirects=False)
    token = response.headers["location"].rsplit("/", 1)[-1]
    client.post(f"/run/{token}", data={
        "mode": "quick", "declare": "1",
        "declared_rarefaction": "10000", "declared_rank": "input",
        "declared_prev_filter": "0.1", "declared_transform": "clr",
        "declared_method": "wilcoxon", "declared_fdr_method": "bh",
        "declared_fdr_threshold": "0.05",
    }, follow_redirects=False)
    assert db.get_job(token).status == "done"

    page = client.get(f"/results/{token}").text
    assert "Your declared pipeline" in page
    assert "percentile" in page

    manifest = client.get(f"/api/jobs/{token}/results").json()["manifest"]
    assert manifest["declared_specification"]["method"] == "wilcoxon"
    assert manifest["declared_percentile"] is not None

    located = client.get(f"/api/jobs/{token}/locate/0").json()
    assert 0.0 <= located["percentile"] <= 100.0


# --- the other two modes run end to end ----------------------------------
def test_full_mode_adds_sophisticated_methods(client):
    response = client.get("/demo/ibd_genus", follow_redirects=False)
    token = response.headers["location"].rsplit("/", 1)[-1]
    client.post(f"/run/{token}", data={"mode": "full"}, follow_redirects=False)
    job = db.get_job(token)
    assert job.status == "done", job.error

    manifest = client.get(f"/api/jobs/{token}/results").json()["manifest"]
    assert manifest["mode"] == "full"
    assert any("sampled" in note for note in manifest["grid"]["notes"])
    frame = client.get(f"/download/{token}/specifications").text
    for method in ("aldex2", "ancombc"):
        assert method in frame


def test_covariate_mode_enumerates_subsets(client):
    response = client.get("/demo/t2d_covariates", follow_redirects=False)
    token = response.headers["location"].rsplit("/", 1)[-1]
    client.post(f"/run/{token}",
                data={"mode": "covariate", "covariates": ["age", "bmi", "sex"]},
                follow_redirects=False)
    job = db.get_job(token)
    assert job.status == "done", job.error

    import pandas as pd

    frame = pd.read_csv(io.BytesIO(client.get(f"/download/{token}/specifications").content))
    subsets = set(frame["covariates"].fillna("").unique())
    assert "" in subsets                       # the null model is always included
    assert "age|bmi|sex" in subsets            # and the full model
    assert len(subsets) == 8                   # 2^3


# --- untrusted values from an uploaded file ------------------------------
def test_a_hostile_taxon_name_is_never_emitted_as_markup(client):
    """Taxon names come from an uploaded file and a results URL is shareable, so an
    unescaped label would be stored XSS against whoever opens the link."""
    payload = "<img src=x onerror=alert(1)>"
    header = "taxon\t" + "\t".join(f"S{i:02d}" for i in range(24)) + "\n"
    rows = [header]
    for t in range(14):
        name = payload if t == 0 else f"Taxon_{t:02d}"
        counts = [str(40 + ((t * 7 + i * 3) % 60)) for i in range(24)]
        rows.append(name + "\t" + "\t".join(counts) + "\n")
    metadata = "sample_id\tgroup\n" + "".join(
        f"S{i:02d}\t{'ctrl' if i < 12 else 'case'}\n" for i in range(24)
    )

    response = client.post(
        "/upload",
        files={"abundance": ("a.tsv", "".join(rows).encode(), "text/tab-separated-values"),
               "metadata": ("m.tsv", metadata.encode(), "text/tab-separated-values")},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text
    token = response.headers["location"].rsplit("/", 1)[-1]
    client.post(f"/run/{token}", data={"mode": "quick"}, follow_redirects=False)
    assert db.get_job(token).status == "done"

    page = client.get(f"/results/{token}").text
    assert payload not in page          # never verbatim
    assert "<img" not in page           # and no tag is ever opened (the page has none)
    # It is present, but only in escaped form: as text, not as markup.
    assert "&lt;img" in page or "\u003cimg" in page

    # The API returns it as data, which is correct — it is a JSON string, not markup.
    taxa = client.get(f"/api/jobs/{token}/results").json()["taxa"]
    assert any(payload in row["taxon"] for row in taxa)

def test_taxon_evidence_page_renders_for_every_tier(client, finished):
    """The page a researcher lands on after clicking a taxon name."""
    table = client.get(f"/api/jobs/{finished}/results").json()["taxa"]
    seen = set()
    for row in table:
        tier = row.get("tier") or row.get("robustness_tier")
        if tier in seen:
            continue
        seen.add(tier)
        response = client.get(f"/results/{finished}/taxon/{row['taxon_id']}")
        assert response.status_code == 200, tier
        body = response.text
        assert "Which choice is responsible?" in body, tier
        assert "Stability, broken apart." in body, tier
        # The limits must travel with the evidence, on every tier.
        assert "not independent evidence" in body, tier
        assert "Nothing here is causal" in body, tier
    assert len(seen) >= 3, "expected several tiers in the demo"


def test_taxon_page_refuses_an_unknown_taxon(client, finished):
    response = client.get(f"/results/{finished}/taxon/999999")
    assert response.status_code == 404
    assert "taxa" in response.text.lower()


def test_taxon_page_never_states_a_probability_of_being_real(client, finished):
    """The share-of-analyses reading is the whole point; it must not drift.

    Checks for an *affirmative* probability claim. The page deliberately contains the
    negated form ("it does not mean a 72% chance the difference is real"), which is the
    disclaimer, so a plain substring search would flag the safeguard as the offence.
    """
    import re

    table = client.get(f"/api/jobs/{finished}/results").json()["taxa"]
    body = client.get(f"/results/{finished}/taxon/{table[0]['taxon_id']}").text
    assert "not probabilities about the biology" in body
    assert "does not mean" in body, "the probability disclaimer is missing"

    lowered = body.lower()
    # "<number>% chance/probability ..." with no negation in the ten words before it.
    affirmative = re.compile(
        r"(?<!not )(?<!never )\d+%\s+(chance|probability)")
    for match in affirmative.finditer(lowered):
        window = lowered[max(0, match.start() - 90):match.start()]
        assert "not " in window or "never" in window, (
            f"affirmative probability claim: ...{lowered[match.start()-70:match.end()+40]}")
    for forbidden in ("independent pieces of evidence", "posterior probability"):
        assert forbidden not in lowered, forbidden


def test_validation_page_renders_the_matrix(client):
    response = client.get("/validation")
    assert response.status_code == 200
    for fragment in ("Empirically validated", "Exploratory",
                     "no held-out experiment",
                     "not validated against an external implementation"):
        assert fragment in response.text, fragment


def test_glossary_terms_reach_the_results_page(client, finished):
    """Inline definitions are how a first-time reader learns the vocabulary."""
    body = client.get(f"/results/{finished}").text
    assert body.count('class="dfn"') >= 5, "too few terms are explained in place"
    assert "dfn-pop" in body

def test_an_unknown_demo_is_a_missing_resource_not_a_bad_request(client):
    """A demo name that does not exist is 404. It was returning 422."""
    response = client.get("/demo/does_not_exist")
    assert response.status_code == 404
    assert "no demo dataset" in response.text.lower()


def test_the_curve_payload_has_one_shape_whether_or_not_it_is_empty(client, finished):
    """The renderer reads data.effect and data.forks directly.

    The empty case used to return {taxon, n_specs, points}, so those came back
    undefined and the plot threw instead of saying there was nothing to plot.
    """
    from app import services
    from app.core.report import specification_curve

    taxa = client.get(f"/api/jobs/{finished}/results").json()["taxa"]
    populated = client.get(
        f"/results/{finished}/curve/{taxa[0]['taxon_id']}").json()

    required = {"taxon", "n_specs", "effect", "significant", "p_adjusted",
                "forks", "fork_labels", "categories", "labels",
                "group_a", "group_b", "declared_position"}
    assert required <= set(populated), required - set(populated)

    _, run, _, _ = services.load_results(finished)
    tested = set(run.long["taxon"].unique())
    untested = next((i for i in range(len(run.taxa_names)) if i not in tested), None)
    if untested is None:
        pytest.skip("every taxon was tested somewhere in this run")
    empty = specification_curve(run, untested)
    assert required <= set(empty), "the empty payload must match the populated one"
    assert empty["n_specs"] == 0
    assert empty["effect"] == []
    assert "nothing to plot" in empty.get("note", "")
