from src.annotate import content_checks as cc


def test_detect_json():
    assert cc.detect_format('{"answer": "x", "items": [1,2,3]}') == "json"
    assert cc.detect_format('[1, 2, 3]') == "json"


def test_detect_markdown_table():
    body = "| key | value |\n| --- | --- |\n| a | 1 |"
    assert cc.detect_format(body) == "markdown_table"


def test_detect_numbered_list():
    body = "1. first\n2. second\n3. third"
    assert cc.detect_format(body) == "numbered_list"


def test_two_item_list_is_not_numbered():
    body = "1. first\n2. second"
    assert cc.detect_format(body) == "plain_paragraph"


def test_detect_plain_paragraph():
    assert cc.detect_format("Just a sentence answering the task.") == "plain_paragraph"


def test_broken_json_falls_through():
    assert cc.detect_format('{"answer": "x", ') == "plain_paragraph"


def test_check_format_match_and_mismatch():
    m = cc.check_format('{"a": 1}', "json")
    assert m["detected_format"] == "json" and m["body_matches_label"] is True
    mm = cc.check_format("plain text", "json")
    assert mm["detected_format"] == "plain_paragraph" and mm["body_matches_label"] is False
