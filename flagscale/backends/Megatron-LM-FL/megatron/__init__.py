# Namespace package configuration
# This allows megatron.core and megatron.plugin to be imported from the installed
# megatron-core package, while other modules (training, rl, post_training, legacy)
# are imported from this source directory.

# Make this a namespace package to allow imports from multiple locations
import sys
import os

# Ensure site-packages are in sys.path before calling pkgutil.extend_path
# This is important for environments where site-packages might not be automatically
# added to sys.path (e.g., when running via torchrun or with custom PYTHONPATH)
try:
    import site
    site_packages = site.getsitepackages()
    for sp in site_packages:
        if sp not in sys.path:
            # Append to sys.path instead of inserting, to maintain PYTHONPATH priority
            sys.path.append(sp)
except Exception:
    # If site.getsitepackages() fails, continue without adding site-packages
    # This can happen in some edge cases
    pass

try:
    import pkgutil
    __path__ = pkgutil.extend_path(__path__, __name__)
except (AttributeError, NameError):
    # If __path__ doesn't exist yet, create it
    __path__ = [os.path.dirname(__file__)]

# The installed megatron-core package will have megatron.core and megatron.plugin
# They will be automatically available through the namespace package mechanism
# No need to explicitly import them here
