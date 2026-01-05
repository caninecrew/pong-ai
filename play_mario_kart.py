"""
Launch Mario Kart 64 through mupen64plus so you can play manually.

Assumptions:
- The MK64 ROM has been copied to the legacy env path:
  mk64-venv/lib/python3.8/site-packages/gym_mupen64plus/ROMs/marioKart.n64
- Mupen64Plus and the Rice video + SDL input plugins are installed
  (see README "Mario Kart 64" section for setup).

Keyboard controls use the default SDL plugin mappings.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent

    # Defaults that match the documented WSL setup.
    rom_path = repo_root / "mk64-venv" / "lib" / "python3.8" / "site-packages" / "gym_mupen64plus" / "ROMs" / "marioKart.n64"
    gfx_plugin = Path("/usr/lib/x86_64-linux-gnu/mupen64plus/mupen64plus-video-rice.so")
    input_driver = Path("/usr/lib/x86_64-linux-gnu/mupen64plus/mupen64plus-input-sdl.so")
    mupen_cmd = os.environ.get("MUPEN_CMD", "mupen64plus")

    # Allow overrides via env vars if needed.
    rom_path = Path(os.environ.get("MK64_ROM", rom_path))
    gfx_plugin = Path(os.environ.get("MK64_GFX_PLUGIN", gfx_plugin))
    input_driver = Path(os.environ.get("MK64_INPUT_DRIVER", input_driver))

    missing = [p for p in (rom_path, gfx_plugin, input_driver) if not p.exists()]
    if missing:
        for p in missing:
            sys.stderr.write(f"Missing file: {p}\n")
        sys.stderr.write("Fix the missing paths or override via MK64_ROM, MK64_GFX_PLUGIN, MK64_INPUT_DRIVER.\n")
        return 1

    cmd = [
        mupen_cmd,
        "--nospeedlimit",
        "--nosaveoptions",
        "--resolution",
        "640x480",
        "--gfx",
        str(gfx_plugin),
        "--audio",
        "dummy",
        "--input",
        str(input_driver),
        str(rom_path),
    ]

    print("Launching mupen64plus. Close the emulator window to exit.")
    print("Command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.stderr.write(f"Could not find mupen64plus executable '{mupen_cmd}'. Is it installed on PATH?\n")
        return 1
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"mupen64plus exited with error code {exc.returncode}\n")
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
