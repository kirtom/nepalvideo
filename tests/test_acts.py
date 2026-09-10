import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from datetime import datetime, timedelta, timezone
import pytest

from nepal.spine.acts import (DayStat, ActBoundary, segment_acts, find_climb_start,
                              act_for, _fit_error)

UTC = timezone.utc
BASE = datetime(2023, 10, 15, tzinfo=UTC)


def day(i, alt):
    s = BASE + timedelta(days=i - 1)
    return DayStat(i, s.date().isoformat(), alt, None if alt is None else alt - 100,
                   s, s + timedelta(hours=23, minutes=59))


# a real Everest Base Camp profile
EBC = [day(1, 2860), day(2, 3440), day(3, 3860), day(4, 4410),
       day(5, 4940), day(6, 5364), day(7, 5545), day(8, 3440)]


def by_act(bounds):
    return {b.act: b for b in bounds}


def test_five_acts_in_order():
    b = segment_acts(EBC)
    assert [x.act for x in b] == [1, 2, 3, 4, 5]


def test_acts_are_contiguous_and_non_decreasing():
    b = segment_acts(EBC)
    for prev, nxt in zip(b, b[1:]):
        assert prev.end_utc == nxt.start_utc
        assert prev.start_utc <= prev.end_utc


def test_act_one_ends_where_the_trek_starts():
    assert by_act(segment_acts(EBC))[1].end_utc == EBC[0].start_utc


def test_act_four_is_the_summit_band_only():
    """Act 4 is 1.5-2 minutes of film -- it must be a narrow band, not a phase."""
    a4 = by_act(segment_acts(EBC))[4]
    assert (a4.end_utc - a4.start_utc) <= timedelta(days=1, hours=1)
    assert a4.start_utc.date() == (BASE + timedelta(days=6)).date()   # day 7


def test_approach_is_shorter_than_the_climb_on_this_profile():
    b = by_act(segment_acts(EBC))
    approach = b[2].end_utc - b[2].start_utc
    climb = b[3].end_utc - b[3].start_utc
    assert climb > approach


def test_climb_starts_where_the_gradient_steepens():
    """Flat for four days, then a hard ascent -- the break must land there."""
    days = [day(1, 2800), day(2, 2820), day(3, 2840), day(4, 2860),
            day(5, 3600), day(6, 4400), day(7, 5200), day(8, 2800)]
    idx = find_climb_start(days, summit_idx=6)
    assert days[idx].day_index in (4, 5), f"got day {days[idx].day_index}"


def test_planning_start_is_honoured():
    start = datetime(2023, 9, 1, tzinfo=UTC)
    assert by_act(segment_acts(EBC, planning_start=start))[1].start_utc == start


def test_after_end_extends_act_five():
    end = datetime(2023, 11, 1, tzinfo=UTC)
    assert by_act(segment_acts(EBC, after_end=end))[5].end_utc == end


# -- degenerate profiles -----------------------------------------------

def test_no_altitude_data_falls_back_to_an_even_split():
    days = [day(i, None) for i in range(1, 9)]
    b = segment_acts(days)
    assert len(b) == 5
    assert all("even-split" in x.method for x in b if x.act > 1)


def test_summit_on_day_one_falls_back():
    days = [day(1, 5545), day(2, 4000), day(3, 3000), day(4, 2860)]
    b = segment_acts(days)
    assert len(b) == 5
    assert any("even-split" in x.method for x in b)


def test_single_day_trek_does_not_produce_inverted_acts():
    b = segment_acts([day(1, 3000)])
    assert len(b) == 5
    for x in b:
        assert x.end_utc >= x.start_utc, f"act {x.act} ends before it starts"


def test_two_day_trek():
    b = segment_acts([day(1, 2860), day(2, 5545)])
    assert len(b) == 5
    for prev, nxt in zip(b, b[1:]):
        assert prev.end_utc == nxt.start_utc


def test_empty_input():
    assert segment_acts([]) == []


def test_monotonic_grind_with_no_breakpoint_still_splits():
    days = [day(i, 2800 + i * 400) for i in range(1, 9)]
    b = segment_acts(days)
    assert len(b) == 5
    assert by_act(b)[3].end_utc > by_act(b)[3].start_utc


# -- lookup ------------------------------------------------------------

def test_act_for_finds_the_containing_act():
    b = segment_acts(EBC)
    assert act_for(BASE + timedelta(days=0, hours=12), b) == 2
    assert act_for(BASE + timedelta(days=6, hours=12), b) == 4


def test_act_for_before_and_after_the_film_clamps():
    b = segment_acts(EBC)
    assert act_for(datetime(2020, 1, 1, tzinfo=UTC), b) == 1
    assert act_for(datetime(2030, 1, 1, tzinfo=UTC), b) == 5


def test_act_for_on_a_shared_boundary_picks_the_opening_act():
    """A shot exactly on a boundary belongs to the act it opens, or the two
    acts would both claim it and the timeline would double-count."""
    b = segment_acts(EBC)
    edge = by_act(b)[3].start_utc
    assert act_for(edge, b) == 3


# -- helper ------------------------------------------------------------

def test_fit_error_on_a_perfect_line():
    err, slope = _fit_error([1, 2, 3, 4], [10, 20, 30, 40])
    assert err == pytest.approx(0.0, abs=1e-9)
    assert slope == pytest.approx(10.0)


def test_fit_error_degenerate_x():
    err, slope = _fit_error([2, 2, 2], [1, 2, 3])
    assert slope == 0.0 and err > 0
