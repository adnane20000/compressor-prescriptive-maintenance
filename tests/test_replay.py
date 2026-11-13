"""Rejeu sans attente, accélération et refus des incohérences temporelles."""

import pytest

from metropt_maintenance.ingestion.replay import iter_replay
from metropt_maintenance.ingestion.schema import SIGNAL_NAMES


def test_replay_timing_and_values_across_chunks(make_csv):
    events = list(iter_replay(make_csv(), chunksize=2, speed=2, max_events=4))
    assert len(events) == 4
    assert [event.delay_seconds for event in events] == [0, 5, 5, 5]
    assert [event.source_delta_seconds for event in events] == [0, 10, 10, 10]
    assert list(events[0].values) == list(SIGNAL_NAMES)
    assert events[0].timestamp < events[-1].timestamp
    assert not any(event.gap_capped for event in events)


def test_date_window_and_explicit_gap_cap(make_csv):
    events = list(iter_replay(
        make_csv(), chunksize=2, start="2020-02-01 00:00:10", end="2020-02-01 00:00:30",
        max_delay_seconds=1,
    ))
    assert len(events) == 3
    assert [event.delay_seconds for event in events] == [0, 1, 1]
    assert events[1].gap_capped and events[1].source_delta_seconds == 10


@pytest.mark.parametrize("chunksize", [1, 2, 10])
def test_replay_refuses_disorder_including_between_chunks(make_csv, chunksize):
    path = make_csv([{}, {"timestamp": "2020-01-31 23:59:59"}])
    with pytest.raises(ValueError, match="non chronologique"):
        list(iter_replay(path, chunksize=chunksize))


def test_replay_rejects_bad_dates_and_preserves_equal_dates(make_csv):
    with pytest.raises(ValueError, match="horodatage"):
        list(iter_replay(make_csv([{"timestamp": "bad"}])))
    events = list(iter_replay(make_csv([{}, {"timestamp": "2020-02-01 00:00:00", "TP2": ""}]), chunksize=1))
    assert events[1].delay_seconds == 0
    assert events[1].values["TP2"] is None


@pytest.mark.parametrize("options", [
    {"speed": 0}, {"speed": float("nan")}, {"max_events": 0},
    {"max_delay_seconds": -1}, {"max_delay_seconds": float("inf")},
    {"start": "2020-02-01", "end": "2020-01-01"},
    {"start": "2020-01-01T00:00:00Z"}, {"start": "NaT"},
])
def test_replay_options_are_validated(make_csv, options):
    with pytest.raises(ValueError):
        list(iter_replay(make_csv(), **options))
