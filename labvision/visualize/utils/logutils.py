import re


__ignoring__ = ['testing#', 'pure#']


def re_match(line):
    try:
        res = re.search(r'\[(?P<epoch>\d+)\,\ +(?P<iter>\d+)\/(?P<total_iter>\d+)]\ +(?P<metrics_name>.*)\:\ +(?P<value>.*)', line)
        return res.groupdict()
    except Exception:
        return None


def readlines(fp, _hash, target=None):
    with open(fp, 'r') as f:
        for line in f.readlines():
            skip = False
            for lbl in __ignoring__:
                if lbl in line:
                    skip = True
            if skip:
                continue
            if _hash in line:
                found = re_match(line)
                if not found:
                    continue
                if not target:
                    yield found
                if target == found['metrics_name']:
                    yield found
