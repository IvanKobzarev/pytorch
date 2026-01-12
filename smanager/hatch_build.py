import subprocess
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        try:
            count = subprocess.check_output(
                ['git', 'rev-list', '--count', 'HEAD'],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
            ver = f"0.{count}"
        except Exception:
            ver = "0.dev"

        # Write _version.py
        version_file = Path(self.root) / "smanager" / "_version.py"
        version_file.write_text(f'__version__ = "{ver}"\n')
