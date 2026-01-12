import subprocess
import sys
from pathlib import Path

# Add our directory to path so get_version can be imported
sys.path.insert(0, str(Path(__file__).parent))

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from get_version import get_git_version


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Write _version.py with the computed version for runtime use
        ver = get_git_version(self.root)
        version_file = Path(self.root) / "smanager" / "_version.py"
        version_file.write_text(f'__version__ = "{ver}"\n')
