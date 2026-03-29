# Nacitani dat z Apple Health
# Martin Silar, SPSE Jecna C4c
# v1 - zakladni nacteni a zobrazeni dat
from lxml import etree
from datetime import datetime
import csv
import os

INPUT_XML = "export.xml"
OUT_DIR = "health_csv"

os.makedirs(OUT_DIR, exist_ok=True)

writers = {}
files = {}


def parse_date(d):
    return datetime.strptime(
        d, "%Y-%m-%d %H:%M:%S %z"
    ).timestamp()


context = etree.iterparse(
    INPUT_XML,
    events=("end",),
    tag="Record"
)

for _, elem in context:
    a = elem.attrib
    record_type = a["type"]
    start = parse_date(a["startDate"])
    end = parse_date(a["endDate"])

    duration = end - start

    try:
        value = float(a.get("value", "nan"))
    except:
        value = a.get("value")

    row = {
        "timestamp_start": start,
        "timestamp_end": end,
        "duration_sec": duration,
        "value": value,
        "unit": a.get("unit"),
        "source": a.get("sourceName"),
        "device": a.get("device")
    }

if record_type not in writers:
    f = open(
        f"{OUT_DIR}/{record_type}.csv",
        "w",
        newline="",
        encoding="utf-8"
    )

    files[record_type] = f

    writers[record_type] = csv.DictWriter(
        f,
        fieldnames=row.keys()
    )
    writers[record_type].writeheader()

writers[record_type].writerow(row)

elem.clear()
while elem.getprevious() is not None:
    del elem.getparent()[0]

for f in files.values():
    f.close()
