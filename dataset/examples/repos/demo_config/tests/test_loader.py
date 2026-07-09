from appconfig.loader import load


def test_legacy_keys_are_normalized():
    assert load({"db_url": "x", "tmo": 5}) == {"database_url": "x", "timeout_sec": 5}


def test_modern_keys_pass_through():
    assert load({"database_url": "y"}) == {"database_url": "y"}
