"""Neon Reef: a display-only recolor of Jumper.

Writes assets/jumper/jumper_neon_reef.xml from assets/jumper/jumper.xml.
Only visual geoms (class="visual") change: they get glowing materials.
Collision geoms, masses, joints and limits are untouched.
"""
import pathlib
import re
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "assets/jumper/jumper.xml"
DST = ROOT / "assets/jumper/jumper_neon_reef.xml"

# name: (rgba, emission, specular, shininess)
MATERIALS = {
    "neon_shell":   ("0.05 0.03 0.10 1", "0.05", "1.0", "1.0"),  # glossy black-violet
    "neon_base":    ("0.07 0.08 0.13 1", "0",    "0.6", "0.7"),
    "neon_cyan":    ("0.0 0.90 1.0 1",   "0.55", "0.5", "0.6"),  # primary limbs
    "neon_magenta": ("1.0 0.17 0.84 1",  "0.55", "0.5", "0.6"),  # forearms / thighs
    "neon_accent":  ("0.82 1.0 0.30 1",  "0.7",  "0.3", "0.4"),  # pads, tips, inserts
}

LIMBS = ("LF_", "RF_", "LM_", "RM_", "LR_", "RR_")


def material_for(geom_name: str) -> str | None:
    name = geom_name.removesuffix("_visual")
    if name == "upper_shell_link":
        return "neon_shell"
    if name == "base_link":
        return "neon_base"
    if name.startswith(LIMBS):
        if re.search(r"pad_|tip_|insert_", name):
            return "neon_accent"
        # alternating bands down each leg: magenta, cyan, magenta, cyan
        if re.search(r"shoulder_|forearm_|thigh_|calf_", name):
            return "neon_magenta"
        return "neon_cyan"
    return None  # display module, sensors, camera keep their original look


def main() -> None:
    tree = ET.parse(SRC)
    root = tree.getroot()
    asset = root.find("asset")
    for name, (rgba, emission, specular, shininess) in MATERIALS.items():
        ET.SubElement(asset, "material", name=name, rgba=rgba, emission=emission,
                      specular=specular, shininess=shininess)
    changed = 0
    for geom in root.iter("geom"):
        if geom.get("class") != "visual":
            continue
        mat = material_for(geom.get("name", ""))
        if mat:
            geom.attrib.pop("rgba", None)
            geom.set("material", mat)
            changed += 1
    root.set("model", "jumper_neon_reef")
    tree.write(DST, encoding="unicode")
    print(f"wrote {DST} ({changed} visual geoms recolored)")


if __name__ == "__main__":
    main()
