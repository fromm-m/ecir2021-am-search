from arclus.utils import is_blank


def test_is_blank():
    """unittest for is_blank"""
    for text in {
        '',
        ' ',
        '\n',
        '   \n',
    }:
        assert is_blank(text=text)
    for text in {
        'a',
        'a    ',
        '     a',
        '\n a',
        '\t a',
    }:
        assert not is_blank(text=text)
