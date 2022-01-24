from datetime import timedelta


def format_timedelta(delta: timedelta, show_microsecond: bool = False) -> str:
    """格式化timedelta

    默认不显示microsecond
    """
    mm, ss = divmod(delta.seconds, 60)
    hh, mm = divmod(mm, 60)
    s = "%d:%02d:%02d" % (hh, mm, ss)
    if delta.days:
        def plural(n):
            return n, abs(n) != 1 and "s" or ""
        s = ("%d day%s, " % plural(delta.days)) + s
    if show_microsecond and delta.microseconds:
        s = s + ".%06d" % delta.microseconds
    return s
