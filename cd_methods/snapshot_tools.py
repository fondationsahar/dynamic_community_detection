from datetime import datetime, timedelta


def get_5_min(ts):
    dt = datetime.fromtimestamp(ts)
    rounded_minutes = (dt.minute // 5) * 5
    quarter_hour_dt = dt.replace(minute=rounded_minutes, second=0, microsecond=0)
    return int(quarter_hour_dt.timestamp())


def get_quarter_hour(ts):
    dt = datetime.fromtimestamp(ts)
    rounded_minutes = (dt.minute // 15) * 15
    quarter_hour_dt = dt.replace(minute=rounded_minutes, second=0, microsecond=0)
    return int(quarter_hour_dt.timestamp())


def get_half_hour(ts):
    dt = datetime.fromtimestamp(ts)
    rounded_minutes = (dt.minute // 30) * 30
    half_hour_dt = dt.replace(minute=rounded_minutes, second=0, microsecond=0)
    return int(half_hour_dt.timestamp())


def get_hour(ts):
    dt = datetime.fromtimestamp(ts)
    rounded_minutes = (dt.minute // 60) * 60
    half_hour_dt = dt.replace(minute=rounded_minutes, second=0, microsecond=0)
    return int(half_hour_dt.timestamp())


def get_half_day(ts):
    dt = datetime.fromtimestamp(ts)
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    noon = midnight + timedelta(hours=12)
    if dt < noon:
        half_day_dt = midnight
    else:
        half_day_dt = noon
    return int(half_day_dt.timestamp())


def get_day(ts):
    dt = datetime.fromtimestamp(ts)
    day_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(day_dt.timestamp())


def group_by_time(timestamps, groupby_type="half_hour"):
    grouped = {}
    for ts in timestamps:
        if groupby_type == "5min":
            section = get_5_min(ts)
        elif groupby_type == "quarter_hour":
            section = get_quarter_hour(ts)
        elif groupby_type == "half_hour":
            section = get_half_hour(ts)
        elif groupby_type == "hour":
            section = get_hour(ts)
        elif groupby_type == "half_day":
            section = get_half_day(ts)
        elif groupby_type == "day":
            section = get_day(ts)
        if section not in grouped:
            grouped[section] = []
        grouped[section].append(ts)

    return grouped
