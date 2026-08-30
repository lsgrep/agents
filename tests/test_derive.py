import pytest

from agentlab.derive import Derivation, Worksheet, fmt


def test_a_derivation_keeps_the_number_as_well_as_the_working():
    # The point of showing the work is not to replace the answer with a
    # printout — a notebook has to be able to keep using the value.
    d = Derivation("t").step("x", "2 * 3", 6, "tok")
    assert d.value == 6 and d.unit == "tok"
    assert "2 * 3" in str(d)


def test_a_failed_check_is_visible_in_the_render():
    d = Derivation("t").check("fits", False, "it does not")
    assert not d.all_checks_passed
    assert "LOOK" in str(d)


def test_givens_carry_where_you_read_them_off():
    d = Derivation("t").given("turns", 40, "turns", source="a real trace")
    assert "a real trace" in str(d)


def test_worksheet_rolls_up_its_parts_checks():
    ws = Worksheet("w")
    ws.add(Derivation("a").check("ok", True))
    assert ws.all_checks_passed
    ws.add(Derivation("b").check("no", False))
    assert not ws.all_checks_passed
    assert "SOME CHECKS" in str(ws)


def test_worksheet_results_are_retrievable_by_name():
    ws = Worksheet("w").add(Derivation("cost of a thing").step("x", "1", 42))
    assert ws.result("cost") == 42
    with pytest.raises(KeyError):
        ws.result("nothing like this")


@pytest.mark.parametrize(
    "value,unit,expected",
    [(0.5, "%", "50.00%"), (1234567, "tok", "1,234,567 tok"), (2.5, "$", "$2.50"), (None, "", "—")],
)
def test_numbers_are_formatted_the_way_you_would_say_them(value, unit, expected):
    assert fmt(value, unit) == expected
