"""Build only the reviewed core, including for editable installs."""
from setuptools import build_meta as _setuptools
from setuptools.build_meta import (
    build_sdist,
    build_wheel,
    get_requires_for_build_editable,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel,
)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    settings = dict(config_settings or {})
    if settings.get("editable_mode", "strict") != "strict":
        raise ValueError("Core editable installs require editable_mode=strict")
    settings["editable_mode"] = "strict"
    return _setuptools.build_editable(wheel_directory, settings, metadata_directory)
