from flame.utils.rich_formatter import rich_format


def test_rich_format():
    a = '123'
    output = rich_format(a)
    assert a in output