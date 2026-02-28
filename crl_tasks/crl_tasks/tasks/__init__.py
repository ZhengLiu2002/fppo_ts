"""Task configurations for CRL IsaacLab."""

from isaaclab_tasks.utils import import_packages

# Import all task subpackages (e.g. locomotion, manipulation).
_BLACKLIST_PKGS: list[str] = []
import_packages(__name__, _BLACKLIST_PKGS)
