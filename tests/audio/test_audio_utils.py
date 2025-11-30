from collections import UserDict

from pi_assistant.audio.processing.utils import device_info_dict


def test_device_info_dict_from_plain_dict():
    data = {"name": "loopback", "index": 2}
    result = device_info_dict(data)
    assert result == data
    assert result is not data  # returns a copy


def test_device_info_dict_from_mapping():
    data = UserDict({"name": "usb mic", "index": 1})
    result = device_info_dict(data)
    assert result == {"name": "usb mic", "index": 1}


def test_device_info_dict_from_object_with_dunder_dict():
    class Info:
        def __init__(self):
            self.name = "embedded"
            self.index = 0

    info = Info()
    result = device_info_dict(info)
    assert result == {"name": "embedded", "index": 0}


def test_device_info_dict_unknown_type_returns_empty():
    class SlotsOnly:
        __slots__ = ("name",)

        def __init__(self):
            self.name = "slots"

    result = device_info_dict(SlotsOnly())
    assert result == {}
