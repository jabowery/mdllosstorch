#!/usr/bin/env python
import sys, pathlib, zipfile, json, re
from email import message_from_string

root = pathlib.Path(".")
dists = sorted(root.glob("dist/*.whl"))
if not dists:
    print("No wheels in dist/")
    sys.exit(0)

wheel = dists[-1]
print(f"Checking wheel: {wheel.name}")
with zipfile.ZipFile(wheel, "r") as z:
    # Find METADATA
    meta_name = next((n for n in z.namelist() if n.endswith("METADATA")), None)
    if not meta_name:
        print("No METADATA found in wheel")
        sys.exit(1)
    meta = z.read(meta_name).decode("utf-8", errors="replace")
    msg = message_from_string(meta)

    name = msg.get("Name")
    version = msg.get("Version")
    requires = msg.get_all("Requires-Dist") or []
    print("Name:", name)
    print("Version:", version)
    print("Requires-Dist:")
    for r in requires:
        print("  -", r)

    # Simple sanity checks
    if name != "mdllosstorch":
        print("ERROR: package name mismatch in wheel METADATA")
        sys.exit(2)
    if not re.match(r"\d+\.\d+\.\d+", version or ""):
        print("ERROR: version not semver-like")
        sys.exit(3)

print("METADATA looks sane.")
