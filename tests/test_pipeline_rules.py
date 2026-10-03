"""Guard rails for the data and feature rules described in CLAUDE.md."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from general_conference_runtime_predictor.data import (
    SPEAKER_ALIASES, _finish, canonical_speaker, parse_runtime,
)
from general_conference_runtime_predictor.features import add_history_features, history_features


def _talks(rows):
    df = pd.DataFrame(rows, columns=["conf_index", "session", "speaker_order", "speaker", "duration_sec"])
    df["duration_status"] = np.where(df.duration_sec.notna(), "ok", "missing")
    df["calling_group"] = "seventy"
    df["role_norm"] = "Of the Seventy"
    df["month"] = "April"
    return df


def test_same_conference_talks_never_enter_each_others_history():
    df = _talks([
        (1, "saturday-morning", 1, "A", 600.0),
        (2, "saturday-morning", 1, "A", 900.0),   # same speaker twice in conference 2
        (2, "sunday-morning", 1, "A", 1200.0),
        (2, "sunday-morning", 2, "B", 700.0),      # B has no earlier history
    ])
    out = add_history_features(df)
    a2 = out[(out.conf_index == 2) & (out.speaker == "A")]
    assert (a2.spk_mean == 600.0).all(), "speaker mean must use only conference 1"
    assert (a2.spk_n_talks == 1).all() and (a2.spk_n_timed == 1).all()
    assert (a2.spk_last == 600.0).all()
    b2 = out[(out.conf_index == 2) & (out.speaker == "B")]
    assert b2.spk_n_talks.item() == 0 and np.isnan(b2.spk_mean.item())
    # Group-level stats for conference 2 come from conference 1 alone.
    assert (out[out.conf_index == 2].global_mean == 600.0).all()
    assert np.isnan(out[out.conf_index == 1].global_mean).all()


def test_history_features_rejects_same_or_later_conference():
    df = _talks([(1, "s", 1, "A", 600.0), (2, "s", 1, "A", 900.0)])
    with pytest.raises(ValueError):
        history_features(df, df[df.conf_index == 2])


def test_untimed_talks_count_as_talks_but_not_as_durations():
    df = _talks([(1, "s", 1, "A", np.nan), (2, "s", 1, "A", 900.0)])
    out = add_history_features(df)
    row = out[out.conf_index == 2].iloc[0]
    assert row.spk_n_talks == 1 and row.spk_n_timed == 0 and np.isnan(row.spk_mean)


def test_mm60_runtimes_are_not_silently_parsed():
    sec, status = parse_runtime("17:60")
    assert np.isnan(sec) and status == "unparseable"
    assert parse_runtime("17:59") == (1079.0, "ok")
    assert parse_runtime(None)[1] == "missing"


def test_aliases_map_to_one_canonical_name():
    assert canonical_speaker("Becky Craven") == "Rebecca L. Craven"
    assert canonical_speaker("Rebecca L. Craven") == "Rebecca L. Craven"
    for src, canon in SPEAKER_ALIASES.items():
        assert canon not in SPEAKER_ALIASES, "canonical names must not themselves be aliased"
        assert src != canon


def test_speaker_order_skips_non_talks_and_renumbers():
    df = pd.DataFrame({
        "year": [2026] * 4, "month_num": [4] * 4, "session": ["saturday-afternoon"] * 4,
        "orig_order": [1, 2, 3, 4],
        "speaker": ["X", "Y", "Z", "W"], "role": ["Of the Seventy"] * 4,
        "title": ["Sustaining of General Authorities, Area Seventies, and General Officers",
                  "Church Auditing Department Report, 2025", "A Talk", "Another Talk"],
        "kicker": [None] * 4, "url": [f"https://x/{i}" for i in range(4)], "num_words": [0, 0, 1500, 1600],
        "duration_raw": [None] * 4, "duration_sec": [np.nan] * 4, "duration_status": ["missing"] * 4,
    })
    out = _finish(df, source="test", duration_source="test")
    assert out.speaker.tolist() == ["Z", "W"]
    assert out.speaker_order.tolist() == [1, 2]
    assert out.attrs["dropped_non_talks"] == 2
