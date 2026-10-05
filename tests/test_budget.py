"""Unit tests for the budget/vendor/naming primitives (pure functions, no DB)."""
from collections import Counter

from fourier.packs.curate import _allocate_budget, _cap_vendor_share, _note_name, _family_kmin


def test_allocate_budget_hits_budget_when_room():
    a = _allocate_budget([100, 100, 100, 100], 120, 6, 40)
    assert sum(a) == 120
    assert all(6 <= x <= 40 for x in a)


def test_allocate_budget_never_exceeds_budget():
    for budget in (50, 100, 300, 1000, 5000):
        a = _allocate_budget([200, 150, 100, 80, 40], budget, 6, 40)
        assert sum(a) <= budget


def test_allocate_budget_capped_by_ceiling_lands_under():
    # 2 families x ceiling 40 = 80 max, budget 500 unreachable -> lands at 80
    a = _allocate_budget([500, 500], 500, 6, 40)
    assert sum(a) == 80
    assert max(a) <= 40


def test_allocate_budget_respects_cluster_size():
    # tiny clusters can't give more than they hold
    a = _allocate_budget([3, 3, 100], 200, 6, 40)
    assert a[0] <= 3 and a[1] <= 3


def test_allocate_budget_deterministic():
    s = [137, 88, 203, 44, 90]
    assert _allocate_budget(s, 150, 6, 40) == _allocate_budget(s, 150, 6, 40)


def test_allocate_budget_proportional():
    # bigger cluster gets at least as many slots as a smaller one
    a = _allocate_budget([300, 100], 400, 6, 200)
    assert a[0] >= a[1]


def test_allocate_budget_empty():
    assert _allocate_budget([], 100, 6, 40) == []


def test_note_name():
    assert _note_name(60) == "C4"
    assert _note_name(61) == "Cs4"
    assert _note_name(36) == "C2"
    assert _note_name(None) is None


def test_cap_vendor_share_caps_dominant():
    # vendor A dominates 8/10; cap 0.5 -> A trimmed to <= the rest (2)
    rec = [{"vendor": "A"} for _ in range(8)] + [{"vendor": "B"} for _ in range(2)]
    out = _cap_vendor_share(rec, None, 0.5, min_vendors=2)
    v = Counter(r["vendor"] for r in out)
    assert v["A"] <= 2
    assert v["A"] / len(out) <= 0.5 + 1e-9
    # three vendors: capped by default
    rec3 = rec + [{"vendor": "C"} for _ in range(2)]
    v3 = Counter(r["vendor"] for r in _cap_vendor_share(rec3, None, 0.5))
    assert v3["A"] <= 4 and v3["B"] == v3["C"] == 2


def _recs(vendors, qual=None):
    return [{"vendor": v, "id": i, "qual": (qual or {}).get(i, 0.5)} for i, v in enumerate(vendors)]


def test_cap_vendor_share_leaves_a_pool_from_one_or_two_vendors_alone():
    # one vendor is all of its pool, and two can't both stay under 40%: no cap, so a small
    # library's only kick pack still fills KICKS
    one = _recs("A" * 6)
    assert _cap_vendor_share(one, None, 0.4) == one
    two = _recs("A" * 8 + "B" * 4)
    assert _cap_vendor_share(two, None, 0.4) == two


def test_cap_vendor_share_never_cuts_below_the_minimum():
    # A 6 of 9 (67%), 0.4 -> A's target round(0.667 * 3) = 2: 5 left, under a minimum of 6
    rec = _recs("AAAAAABCD", qual={0: .1, 1: .9, 2: .8, 3: .2, 4: .7, 5: .3})
    capped = _cap_vendor_share(rec, None, 0.4)
    assert len(capped) == 5
    kept = _cap_vendor_share(rec, None, 0.4, keep=6)
    assert len(kept) == 6 and [r["id"] for r in kept] == sorted(r["id"] for r in kept)   # rec's order
    back = {r["id"] for r in kept} - {r["id"] for r in capped}
    assert len(back) == 1 and max(rec[i]["qual"] for i in back) >= max(
        r["qual"] for r in rec[:6] if r["id"] not in {x["id"] for x in kept})   # the best came back


def test_cap_vendor_share_noop_when_balanced():
    rec = [{"vendor": "A"} for _ in range(5)] + [{"vendor": "B"} for _ in range(5)]
    out = _cap_vendor_share(rec, None, 0.5)
    assert len(out) == 10


def test_cap_vendor_share_deterministic():
    rec = [{"vendor": "A"} for _ in range(20)] + [{"vendor": "B"} for _ in range(5)]
    assert [r["vendor"] for r in _cap_vendor_share(rec, None, 0.5)] == \
           [r["vendor"] for r in _cap_vendor_share(rec, None, 0.5)]


def test_family_kmin_leaves_room_under_the_ceiling():
    # KICKS: 550 over 60-file folders needs >= 14 families so size still matters
    assert _family_kmin(550, 60, 8, 30) == 14
    assert _family_kmin(100, 60, 8, 30) == 8          # small budget: kmin stands
    assert _family_kmin(5000, 60, 8, 30) == 30        # never past kmax
    assert _family_kmin(550, 60, 8, 30, n=60, floor=6) == 10   # only as many as can fill
    assert _family_kmin(550, 60, 8, 30, n=10, floor=6) == 8    # ...never below kmin


def test_more_families_make_allocation_proportional_again():
    sizes = [1277, 1243, 996, 946, 882, 635, 581, 352, 186]     # KICKS pools (9 families)
    flat = _allocate_budget(sizes, 550, 6, 60)
    assert len(set(flat)) == 1                                   # every folder at the ceiling
    split = [s // 2 for s in sizes] + [s - s // 2 for s in sizes][:5]   # 14 families
    a = _allocate_budget(split, 550, 6, 60)
    assert sum(a) == 550 and max(a) <= 60 and len(set(a)) > 3
