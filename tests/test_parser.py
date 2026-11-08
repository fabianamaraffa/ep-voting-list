from ep_voting.parser import parse_pdf, strip_markers


def by_number(result):
    return {a.number: a for a in result.amendments}


def test_fixture_finds_all_amendments_without_warnings(fixture_pdf):
    result = parse_pdf(fixture_pdf)
    assert [a.number for a in result.amendments] == [1, 2, 3, 4, 5, 6]
    assert result.expected_range == (1, 6)
    assert result.warnings == []


def test_new_insertion_has_empty_left_column(fixture_pdf):
    a = by_number(parse_pdf(fixture_pdf))[1]
    assert a.is_new and a.original_text == ""
    assert "Youth Guarantee" in a.amendment_text
    assert "**" not in a.amendment_text  # fully new text needs no change markers
    assert a.location == "Citation 3 a (new)"
    assert a.doc_type == "Motion for a resolution"


def test_changes_are_marked_and_columns_do_not_mix(fixture_pdf):
    a = by_number(parse_pdf(fixture_pdf))[2]
    assert a.original_text == "3. Calls for support **existing** instruments;"
    assert a.amendment_text == "3. Calls for support **new youth** instruments;"


def test_heading_separated_from_authors_by_page_break(fixture_pdf):
    a = by_number(parse_pdf(fixture_pdf))[3]
    assert a.authors == "Dora Mid, Emil Long-Name Third"
    assert a.location == "Paragraph 5"
    assert a.language == "de"
    assert a.page == 3


def test_hyphenated_line_wrap_is_rejoined(fixture_pdf):
    a = by_number(parse_pdf(fixture_pdf))[3]
    assert "stand-alone nature" in a.original_text


def test_table_split_across_pages_and_justification(fixture_pdf):
    a = by_number(parse_pdf(fixture_pdf))[4]
    assert strip_markers(a.amendment_text).endswith("third line of text;")
    assert "with change" in a.amendment_text
    assert a.justification == "Funding must stay with youth organisations."


def test_in_text_mention_of_amendment_does_not_split(fixture_pdf):
    result = parse_pdf(fixture_pdf)
    assert len(result.amendments) == 6
    a = by_number(result)[5]
    assert "Amendment 5" in a.original_text and "Amendment 5" in a.amendment_text


def test_deletion_and_language(fixture_pdf):
    a = by_number(parse_pdf(fixture_pdf))[6]
    assert a.amendment_text == "deleted"
    assert a.language == "hu"


def test_missing_amendments_are_reported(fixture_pdf):
    result = parse_pdf(fixture_pdf, max_pages=4)  # stops before AM 4-6
    assert 6 not in {a.number for a in result.amendments}


# ----- real document (only when EP_SAMPLE_PDF is set) -----------------------------

def test_real_document_is_fully_parsed(real_pdf):
    result = parse_pdf(real_pdf)
    assert result.expected_range == (1, 317)
    assert [a.number for a in result.amendments] == list(range(1, 318))
    assert result.warnings == []
    assert all(a.authors and a.location and a.amendment_text for a in result.amendments)


def test_real_document_known_amendments(real_pdf):
    amendments = by_number(parse_pdf(real_pdf))
    assert amendments[1].is_new
    assert amendments[86].authors.startswith("Brigitte van den Berg")  # heading and authors on different pages
    assert amendments[87].authors == "Miriam Lexmann, Jagna Marczułajtis-Walczak"
    assert "the aim of" in amendments[87].amendment_text
    assert amendments[85].language == "el"
