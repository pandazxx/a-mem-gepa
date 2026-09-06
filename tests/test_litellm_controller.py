from amem_gepa.llm.litellm_controller import parse_json_response


def test_parse_json_response_plain():
    assert parse_json_response('{"a": 1}') == {"a": 1}


def test_parse_json_response_strips_markdown_fence():
    raw = '```json\n{"a": 1}\n```'
    assert parse_json_response(raw) == {"a": 1}
