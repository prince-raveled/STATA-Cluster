"""Shared Jinja2 environment and template filters."""
from __future__ import annotations

import numpy as np
from fastapi.templating import Jinja2Templates

from . import config
from .core.evidence import (
    CROSS_STUDY,
    GRADE_BLURB,
    GRADE_LABEL,
    NEARING_STUDY,
    PUBLISHED_STUDY,
    TIER_CAVEATS,
    TIER_REPLICATION,
    TIER_VALIDATION,
    V8_PREREGISTRATION,
    WEIGHTING_VALIDATION,
    matrix_rows,
    tier_sentence,
)
from .core.glossary import glossary_sections
from .core.models import FORK_LABELS, METHOD_LABELS, ordinal
from .core.robustness import TIERS
from .core.validity import defensibility_register, rule_id
from .ui import (
    MODE_SUMMARIES,
    TIER_HEADLINES,
    glossary_with_ui_terms,
    help_term,
    run_stages,
    run_time,
    skipped_reasons,
    workflow_steps,
)

templates = Jinja2Templates(directory=str(config.BASE_DIR / "app" / "templates"))


def commafy(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def percent(value, places: int = 0) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    return f"{number:.{places}%}"


def signed(value, places: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    return f"{number:+.{places}f}"


def fixed(value, places: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    return f"{number:.{places}f}"


def duration(seconds) -> str:
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return "-"
    if total < 60:
        return f"{total:.1f} s"
    minutes, rest = divmod(total, 60)
    return f"{int(minutes)} min {rest:.0f} s"


templates.env.filters["commafy"] = commafy
templates.env.filters["percent"] = percent
templates.env.filters["signed"] = signed
templates.env.filters["fixed"] = fixed
templates.env.filters["duration"] = duration
templates.env.filters["ordinal"] = ordinal
templates.env.filters["tier_sentence"] = tier_sentence
templates.env.globals.update(
    TERM=help_term,
    GLOSSARY=glossary_with_ui_terms(glossary_sections()),
    TIER_HEADLINES=TIER_HEADLINES,
    MODE_SUMMARIES=MODE_SUMMARIES,
    TIER_ORDER=list(TIERS),
    WORKFLOW_STEPS=workflow_steps,
    RUN_STAGES=run_stages,
    RUN_TIME=run_time,
    SKIPPED_REASONS=skipped_reasons,
    # Changes whenever the stylesheet or scripts change shape, so a returning visitor
    # never renders new markup against a cached stylesheet. The application version
    # is recorded in every run manifest and is deliberately not bumped for a restyle.
    ASSETS=f"{config.VERSION}-ui7",
    TIERS=TIERS,
    TIER_REPLICATION=TIER_REPLICATION,
    TIER_VALIDATION=TIER_VALIDATION,
    V8=V8_PREREGISTRATION,
    V8_RECORD=WEIGHTING_VALIDATION,
    TIER_CAVEATS=TIER_CAVEATS,
    PUBLISHED_STUDY=PUBLISHED_STUDY,
    CROSS_STUDY=CROSS_STUDY,
    NEARING_STUDY=NEARING_STUDY,
    GRADE_LABEL=GRADE_LABEL,
    GRADE_BLURB=GRADE_BLURB,
    VALIDATION_MATRIX=matrix_rows(),
    FORK_LABELS=FORK_LABELS,
    METHOD_LABELS=METHOD_LABELS,
    VERSION=config.VERSION,
    SPEC_VERSION=config.SPEC_VERSION,
    REPOSITORY=config.REPOSITORY,
    SOURCE_URL=config.source_url,
    RULE_ID=rule_id,
    DEFENSIBILITY=defensibility_register(),
    MODE_LABELS=config.MODE_LABELS,
    MODE_BLURBS=config.MODE_BLURBS,
    RETENTION_DAYS=config.RETENTION_DAYS,
)
