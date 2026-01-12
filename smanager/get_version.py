import subprocess

def get_git_version(root='.'):
    """Get version from git commit count."""
    try:
        count = subprocess.check_output(
            ['git', 'rev-list', '--count', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL,
            cwd=root
        ).strip()
        return f"0.{count}"
    except Exception:
        return "0.dev"
