LEGACY_KEYS = {"db_url": "database_url", "tmo": "timeout_sec"}


def load(pairs: dict) -> dict:
    config = {}
    for key, value in pairs.items():
        config[key] = value
    return config
