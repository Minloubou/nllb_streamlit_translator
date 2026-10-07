from src.utils.config import load_config


def test_config_loads():
    config = load_config("configs/config.yaml")

    assert "model" in config
    assert "generation" in config
    assert "languages" in config
    assert isinstance(config["languages"], dict)
