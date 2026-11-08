from ep_voting.diff import describe_change
from ep_voting.schemas import Amendment


def am(original, amended):
    return Amendment(number=1, original_text=original, amendment_text=amended, is_new=not original)


def test_replacement_gives_removed_and_added():
    removed, added = describe_change(am("education for all, **public** education", "education for all, **accessible vocational** education"))
    assert removed == "public" and added == "accessible vocational"


def test_pure_deletion_of_words():
    removed, added = describe_change(am("calls for **a ban on unpaid** traineeships", "calls for traineeships"))
    assert removed == "a ban on unpaid" and added == ""


def test_pure_addition_of_words():
    removed, added = describe_change(am("calls for traineeships", "calls for **paid** traineeships"))
    assert removed == "" and added == "paid"


def test_new_insertion_has_nothing_removed():
    removed, added = describe_change(am("", "Ba. whereas new text;"))
    assert removed == "" and added == "Ba. whereas new text;"


def test_whole_text_deleted():
    removed, added = describe_change(am("8. Text to be removed;", "deleted"))
    assert removed == "8. Text to be removed;" and added == ""


def test_several_separate_changes_are_joined():
    removed, added = describe_change(am("a x b y c", "a p b q c"))
    assert removed == "x | y" and added == "p | q"


def test_official_marks_are_used_when_present_and_diff_is_the_fallback():
    # Marked: taken from the bold-italic runs, even if a word-level diff would split differently.
    removed, added = describe_change(am("pay **rates** apply", "pay **rates and fees** apply"))
    assert removed == "rates" and added == "rates and fees"
    # Not marked: falls back to the diff.
    removed, added = describe_change(am("pay rates apply", "pay fees apply"))
    assert removed == "rates" and added == "fees"
