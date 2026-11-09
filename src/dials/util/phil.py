from __future__ import annotations

import collections
import contextlib
import os
import re

import libtbx.phil
from dxtbx.model.experiment_list import ExperimentListFactory
from libtbx.utils import Sorry

from dials.array_family import flex

FilenameDataWrapper = collections.namedtuple("FilenameDataWrapper", "filename, data")

# Already-loaded input files, by (phil type, filename), that the converters use
# instead of loading the files again. Set with reuse_loaded_inputs.
_loaded_inputs = {}


@contextlib.contextmanager
def reuse_loaded_inputs(wrappers):
    """
    Within this context, converting the experiment list and reflection table
    parameters with the given FilenameDataWrapper values from phil returns
    those values, rather than loading the files again.

    :param wrappers: A dict of FilenameDataWrapper values by phil type
    """
    global _loaded_inputs
    previous = _loaded_inputs
    _loaded_inputs = {
        (phil_type, wrapper.filename): wrapper
        for phil_type, values in wrappers.items()
        for wrapper in values
    }
    try:
        yield
    finally:
        _loaded_inputs = previous


class ExperimentListConverters:
    """A phil converter for the experiment list class."""

    phil_type = "experiment_list"

    def __init__(self, check_format=True):
        self._check_format = check_format

    def __str__(self):
        return self.phil_type

    def from_words(self, words, master):
        s = libtbx.phil.str_from_words(words=words)
        if s is None:
            return None
        if s == "<image files>":
            return FilenameDataWrapper(filename=s, data=None)
        if (self.phil_type, s) in _loaded_inputs:
            return _loaded_inputs[(self.phil_type, s)]
        if not os.path.exists(s):
            raise Sorry(f"File {s} does not exist")
        return FilenameDataWrapper(
            filename=s,
            data=ExperimentListFactory.from_json_file(
                s, check_format=self._check_format
            ),
        )

    def as_words(self, python_object, master):
        if python_object is None:
            value = "None"
        else:
            value = python_object.filename
        return [libtbx.phil.tokenizer.word(value=value)]


class ReflectionTableConverters:
    """A phil converter for the reflection table class."""

    phil_type = "reflection_table"

    def __str__(self):
        return self.phil_type

    def from_words(self, words, master):
        s = libtbx.phil.str_from_words(words=words)
        if s is None:
            return None
        if (self.phil_type, s) in _loaded_inputs:
            return _loaded_inputs[(self.phil_type, s)]
        if not os.path.exists(s):
            raise Sorry(f"File {s} does not exist")
        return FilenameDataWrapper(filename=s, data=flex.reflection_table.from_file(s))

    def as_words(self, python_object, master):
        if python_object is None:
            value = "None"
        else:
            value = python_object.filename
        return [libtbx.phil.tokenizer.word(value=value)]


class ReflectionTableSelectorConverters:
    """A phil converter for the reflection table selector class."""

    phil_type = "reflection_table_selector"

    def __str__(self):
        return self.phil_type

    def from_words(self, words, master):
        s = libtbx.phil.str_from_words(words=words)
        if s is None:
            return None
        regex = r"^\s*([\w\.]+)\s*(<=|!=|==|>=|<|>|&)\s*(.+)\s*$"
        matches = re.findall(regex, s)
        if len(matches) == 0:
            raise RuntimeError(f"{s} is not a valid selection")
        elif len(matches) > 1:
            raise RuntimeError(f"{s} is not a single selection")
        else:
            match = matches[0]
        assert len(match) == 3
        col, op, value = match
        return flex.reflection_table_selector(col, op, value)

    def as_words(self, python_object, master):
        if python_object is None:
            value = "None"
        else:
            value = (
                f"{python_object.column}{python_object.op_string}{python_object.value}"
            )
        return [libtbx.phil.tokenizer.word(value=value)]


# Create the default converter registry with the extract converters
default_converter_registry = libtbx.phil.extended_converter_registry(
    additional_converters=[
        ExperimentListConverters,
        ReflectionTableConverters,
        ReflectionTableSelectorConverters,
    ]
)


def parse(
    input_string=None,
    source_info=None,
    file_name=None,
    converter_registry=None,
    process_includes=False,
):
    """Redefinition of the parse function."""
    if converter_registry is None:
        converter_registry = default_converter_registry
    return libtbx.phil.parse(
        input_string=input_string,
        source_info=source_info,
        file_name=file_name,
        converter_registry=converter_registry,
        process_includes=process_includes,
    )
