"""读取宿主机及容器内存；冷文件缓存可回收，不能把宿主机余量当容器余量。"""
from pathlib import Path
import psutil

MIB = 1024 ** 2

def memory_report(base=Path("/sys/fs/cgroup"), host_bytes=None):
    host = psutil.virtual_memory().available if host_bytes is None else host_bytes
    result = {"host_physical_available_mib": host / MIB, "host_available_mib": host / MIB}
    limit_file, current_file = base / "memory.max", base / "memory.current"
    if not limit_file.exists() or not current_file.exists():
        return result
    value = limit_file.read_text().strip()
    if value == "max":
        return result
    limit, current = int(value), int(current_file.read_text())
    stat_file = base / "memory.stat"
    stats = dict(line.split() for line in stat_file.read_text().splitlines()) if stat_file.exists() else {}
    inactive = int(stats.get("inactive_file", 0))
    working = max(0, current - inactive)
    available = max(0, min(limit, limit - working))
    result.update(cgroup_limit_mib=limit / MIB, cgroup_current_mib=current / MIB,
                  cgroup_inactive_file_mib=inactive / MIB, cgroup_working_set_mib=working / MIB,
                  host_available_mib=min(host, available) / MIB)
    return result
