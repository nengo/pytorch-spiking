"""
We use semantic versioning (see http://semver.org/).
and conform to PEP440 (see https://www.python.org/dev/peps/pep-0440/).
'.devN' will be added to the version unless the code base represents
a release version. Release versions are git tagged with the version.
"""

name = "pytorch-spiking"
version_info = (0, 1, 0)  # (major, minor, patch)
dev = None  # set to None for releases

version = "{v}{dev}".format(
    v=".".join(str(v) for v in version_info),
    dev=(".dev%d" % dev) if dev is not None else "",
)
