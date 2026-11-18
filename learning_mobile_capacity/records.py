import csv
from pathlib import Path


def read_records(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return []
    required = {"period", "group", "value"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"缺少字段: {', '.join(sorted(missing))}")
    if "count" not in rows[0]:
        # 需求历史每期一条记录时可省略 count，默认按一条记录处理
        for row in rows:
            row.setdefault("count", "1")
    return rows
