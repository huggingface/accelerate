import os
from unittest.mock import patch

from accelerate.utils.launch import env_var_path_add


def test_appends_to_empty_variable():
    with patch.dict(os.environ, {"PYTHONPATH": ""}, clear=False):
        assert env_var_path_add("PYTHONPATH", "/new") == "/new"


def test_appends_to_existing_entries_using_pathsep():
    existing = os.pathsep.join(["/already/there", "/also"])
    with patch.dict(os.environ, {"PYTHONPATH": existing}, clear=False):
        result = env_var_path_add("PYTHONPATH", "/new")
    assert result.split(os.pathsep) == ["/already/there", "/also", "/new"]


def test_windows_pythonpath_is_not_split_on_drive_colon():
    """A Windows PYTHONPATH uses ';' and contains drive-letter colons."""
    existing = r"C:\Users\me\proj;C:\libs"
    with patch.dict(os.environ, {"PYTHONPATH": existing}, clear=False), patch("os.pathsep", ";"):
        result = env_var_path_add("PYTHONPATH", r"D:\cwd")
    assert result.split(";") == [r"C:\Users\me\proj", r"C:\libs", r"D:\cwd"]
