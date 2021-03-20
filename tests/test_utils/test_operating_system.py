from flame.utils import operating_system


def test_find_free_port():
    port = operating_system.find_free_port()
    assert port > 0
