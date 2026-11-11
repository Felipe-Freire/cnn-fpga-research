"""Parsers for Vitis HLS csynth.xml and Vivado utilization / timing reports."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional


def _num(x: Optional[str]):
    if x is None:
        return None
    x = x.strip()
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


def parse_csynth_xml(path: Path) -> Dict:
    root = ET.parse(path).getroot()
    perf = root.find("PerformanceEstimates")
    lat = perf.find("SummaryOfOverallLatency")
    res = root.find("AreaEstimates/Resources")
    avail = root.find("AreaEstimates/AvailableResources")
    out = {
        "estimated_clock_ns": _num(perf.findtext("SummaryOfTimingAnalysis/EstimatedClockPeriod")),
        "latency_best_cycles": _num(lat.findtext("Best-caseLatency")),
        "latency_worst_cycles": _num(lat.findtext("Worst-caseLatency")),
        "interval_min_cycles": _num(lat.findtext("Interval-min")),
        "interval_max_cycles": _num(lat.findtext("Interval-max")),
        "resources": {c.tag: _num(c.text) for c in res} if res is not None else {},
        "available": {c.tag: _num(c.text) for c in avail} if avail is not None else {},
    }
    return out


_UTIL_ROWS = {
    "LUT": r"CLB LUTs\*?",
    "LUT_logic": r"LUT as Logic",
    "LUT_memory": r"LUT as Memory",
    "FF": r"CLB Registers",
    "CARRY8": r"CARRY8",
    "BRAM_tile": r"Block RAM Tile",
    "RAMB36": r"RAMB36/FIFO\*?",
    "RAMB18": r"RAMB18",
    "URAM": r"URAM",
    "DSP": r"DSPs",
}


def parse_vivado_util(path: Path) -> Dict[str, float]:
    text = Path(path).read_text()
    out = {}
    for key, label in _UTIL_ROWS.items():
        m = re.search(r"^\|\s*" + label + r"\s*\|\s*([\d.]+)\s*\|", text, flags=re.M)
        if m:
            out[key] = _num(m.group(1))
    return out


def parse_vivado_util_hier(path: Path) -> List[Dict]:
    """Rows of the hierarchical utilization table: instance, module, depth, LUT, FF, RAMB36, RAMB18, DSP."""
    lines = Path(path).read_text().splitlines()
    header, rows = None, []
    for ln in lines:
        if not ln.startswith("|"):
            continue
        cells = [c for c in ln.split("|")[1:-1]]
        if not cells:
            continue
        if header is None and "Instance" in cells[0]:
            header = [c.strip() for c in cells]
            continue
        if header is None or len(cells) != len(header):
            continue
        inst_raw = cells[0]
        depth = (len(inst_raw) - len(inst_raw.lstrip(" "))) // 2
        rec = {"instance": inst_raw.strip(), "depth": depth}
        for h, c in zip(header[1:], cells[1:]):
            rec[h] = _num(c)
        rows.append(rec)
    return rows


def parse_wns(path: Path) -> Optional[float]:
    text = Path(path).read_text()
    m = re.search(r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*-+.*?\n\s*(-?[\d.]+)", text)
    return float(m.group(1)) if m else None
