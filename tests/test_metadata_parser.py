from src.annotate import metadata_parser as mp

C2_AXES = [{"name": "format", "field": "format",
            "values": ["json", "markdown_table", "numbered_list", "plain_paragraph"]}]
C1_AXES = [
    {"name": "gender_identity", "field": "gender_identity",
     "values": ["woman", "man", "nonbinary"]},
    {"name": "home_region", "field": "home_region",
     "values": ["Africa", "East Asia", "Europe", "Latin America"]},
    {"name": "field", "field": "field", "values": []},
]


def test_single_field_valid():
    out = mp.parse_metadata("Format: json\n\n{...}", "Format", C2_AXES)
    assert out["valid_metadata"] is True
    assert out["parsed_labels"]["format"] == "json"


def test_single_field_alias_and_case_insensitive():
    out = mp.parse_metadata("Format: Table\n\n| a | b |", "Format", C2_AXES)
    assert out["parsed_labels"]["format"] == "markdown_table"


def test_single_field_out_of_set_is_unknown_but_still_valid_line():
    out = mp.parse_metadata("Format: yaml\n\nfoo: bar", "Format", C2_AXES)
    assert out["valid_metadata"] is True
    assert out["parsed_labels"]["format"] == "unknown"


def test_missing_line_is_invalid():
    out = mp.parse_metadata("just an answer with no metadata", "Format", C2_AXES)
    assert out["valid_metadata"] is False
    assert out["parsed_labels"]["format"] == "unknown"


def test_malformed_separator_is_invalid():
    out = mp.parse_metadata("Format = json\n\nbody", "Format", C2_AXES)
    assert out["valid_metadata"] is False


def test_empty_value_is_invalid():
    out = mp.parse_metadata("Format:\n\nbody", "Format", C2_AXES)
    assert out["valid_metadata"] is False


def test_multi_field_valid():
    line = "Profile: gender_identity=woman; home_region=East Asia; field=glaciology"
    out = mp.parse_metadata(line + "\n\nbio...", "Profile", C1_AXES)
    assert out["valid_metadata"] is True
    pl = out["parsed_labels"]
    assert pl["gender_identity"] == "woman"
    assert pl["home_region"] == "East Asia"
    assert pl["field"] == "glaciology"  # descriptive axis keeps raw value


def test_multi_field_partial_missing_subfield_unknown():
    line = "Profile: gender_identity=man"
    out = mp.parse_metadata(line + "\n\nbio", "Profile", C1_AXES)
    assert out["valid_metadata"] is True
    assert out["parsed_labels"]["gender_identity"] == "man"
    assert out["parsed_labels"]["home_region"] == "unknown"


def test_multi_field_no_pairs_is_invalid():
    out = mp.parse_metadata("Profile: a fictional person\n\nbio", "Profile", C1_AXES)
    assert out["valid_metadata"] is False


def test_split_meta_and_body():
    meta, body = mp.split_meta_and_body("Format: json\n\n{\"x\": 1}", "Format")
    assert meta == "Format: json"
    assert body == '{"x": 1}'
    meta2, body2 = mp.split_meta_and_body("no meta here\nsecond line", "Format")
    assert meta2 is None
    assert "no meta here" in body2
