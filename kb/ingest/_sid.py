"""PSID/RSID header parsing for SID music files (HVSC ingestion).

Field offsets, sizes, byte order and the flags bitfield below are taken from
the published SID file format specification shipped with HVSC
(DOCUMENTS/SID_file_format.txt), not inferred from a sample file. The spec's
LEGEND defines WORD as "16-bit big endian encoded binary value" and LONGWORD
as "32-bit big endian encoded binary value", which is why every struct format
here is big-endian - a little-endian read yields plausible-looking garbage
rather than an obvious failure, so it is worth stating.

The header is 118 bytes ($76) for version 1 and 124 bytes ($7C) for versions
2-4; the extra six bytes are flags, startPage, pageLength, secondSIDAddress
and thirdSIDAddress.
"""

import struct
import zipfile
from typing import Any

MAGIC_PSID = b"PSID"
MAGIC_RSID = b"RSID"

# Offsets are the spec's own, in hex, so they can be diffed against its table.
_OFF_MAGIC = 0x00
_OFF_VERSION = 0x04
_OFF_DATA_OFFSET = 0x06
_OFF_LOAD_ADDRESS = 0x08
_OFF_INIT_ADDRESS = 0x0A
_OFF_PLAY_ADDRESS = 0x0C
_OFF_SONGS = 0x0E
_OFF_START_SONG = 0x10
_OFF_SPEED = 0x12
_OFF_NAME = 0x16
_OFF_AUTHOR = 0x36
_OFF_RELEASED = 0x56
_OFF_FLAGS = 0x76
_OFF_START_PAGE = 0x78
_OFF_PAGE_LENGTH = 0x79
_OFF_SECOND_SID = 0x7A
_OFF_THIRD_SID = 0x7B

_STRING_FIELD_LEN = 32
V1_HEADER_SIZE = 0x76   # 118
V2_HEADER_SIZE = 0x7C   # 124

# Spec: "Extended ASCII encoded (Windows-1252 code page)".
_STRING_ENCODING = "cp1252"

# flags bits 4-5 (primary SID), 6-7 (second SID, v3+), 8-9 (third SID, v4+).
_SID_MODEL = {0: "Unknown", 1: "MOS6581", 2: "MOS8580", 3: "MOS6581 and MOS8580"}
# flags bits 2-3.
_CLOCK = {0: "Unknown", 1: "PAL", 2: "NTSC", 3: "PAL and NTSC"}


class SidHeaderError(ValueError):
    """Raised when a file is not a well-formed PSID/RSID."""


def _text_field(raw: bytes) -> str:
    """Decode one 32-byte header string.

    Spec: zero-terminated if shorter than 32 characters, so everything from
    the first NUL onward is padding rather than content.
    """
    nul = raw.find(b"\x00")
    if nul != -1:
        raw = raw[:nul]
    return raw.decode(_STRING_ENCODING, errors="replace").strip()


def parse_sid_header(data: bytes) -> dict[str, Any]:
    """Parse a PSID/RSID header into a plain dict.

    Raises SidHeaderError when the magic, version or length are outside what
    the spec allows, rather than returning half-populated metadata.
    """
    if len(data) < V1_HEADER_SIZE:
        raise SidHeaderError(
            f"file is {len(data)} bytes, shorter than the {V1_HEADER_SIZE}-byte v1 header"
        )

    magic = data[_OFF_MAGIC:_OFF_MAGIC + 4]
    if magic not in (MAGIC_PSID, MAGIC_RSID):
        raise SidHeaderError(f"bad magic {magic!r}: expected b'PSID' or b'RSID'")

    version = struct.unpack_from(">H", data, _OFF_VERSION)[0]
    if version not in (1, 2, 3, 4):
        raise SidHeaderError(f"unsupported version {version}: the spec allows 1-4")

    header_size = V1_HEADER_SIZE if version == 1 else V2_HEADER_SIZE
    if len(data) < header_size:
        raise SidHeaderError(
            f"file is {len(data)} bytes, shorter than the {header_size}-byte v{version} header"
        )

    meta: dict[str, Any] = {
        "format": magic.decode("ascii"),
        "version": version,
        "data_offset": struct.unpack_from(">H", data, _OFF_DATA_OFFSET)[0],
        "load_address": struct.unpack_from(">H", data, _OFF_LOAD_ADDRESS)[0],
        "init_address": struct.unpack_from(">H", data, _OFF_INIT_ADDRESS)[0],
        "play_address": struct.unpack_from(">H", data, _OFF_PLAY_ADDRESS)[0],
        "songs": struct.unpack_from(">H", data, _OFF_SONGS)[0],
        "start_song": struct.unpack_from(">H", data, _OFF_START_SONG)[0],
        "speed": struct.unpack_from(">I", data, _OFF_SPEED)[0],
        "title": _text_field(data[_OFF_NAME:_OFF_NAME + _STRING_FIELD_LEN]),
        "author": _text_field(data[_OFF_AUTHOR:_OFF_AUTHOR + _STRING_FIELD_LEN]),
        "released": _text_field(data[_OFF_RELEASED:_OFF_RELEASED + _STRING_FIELD_LEN]),
    }

    # Spec: loadAddress 0 means the real address is the first two bytes of the
    # data, little-endian - the one little-endian value in an otherwise
    # big-endian format.
    if meta["load_address"] == 0:
        off = meta["data_offset"]
        if len(data) >= off + 2:
            meta["load_address"] = struct.unpack_from("<H", data, off)[0]
            meta["load_address_from_data"] = True

    if version >= 2:
        flags = struct.unpack_from(">H", data, _OFF_FLAGS)[0]
        meta["flags"] = flags
        meta["mus_player"] = bool(flags & 0b1)
        meta["psid_specific"] = bool(flags & 0b10)
        meta["clock"] = _CLOCK[(flags >> 2) & 0b11]
        meta["chip_model"] = _SID_MODEL[(flags >> 4) & 0b11]
        if version >= 3:
            meta["second_chip_model"] = _SID_MODEL[(flags >> 6) & 0b11]
            meta["second_sid_address"] = data[_OFF_SECOND_SID]
        if version >= 4:
            meta["third_chip_model"] = _SID_MODEL[(flags >> 8) & 0b11]
            meta["third_sid_address"] = data[_OFF_THIRD_SID]
        meta["start_page"] = data[_OFF_START_PAGE]
        meta["page_length"] = data[_OFF_PAGE_LENGTH]
    else:
        # v1 has no flags word at all. Saying "Unknown" is honest; inventing a
        # chip model because most tunes are 6581 is not.
        meta["chip_model"] = "Unknown"
        meta["clock"] = "Unknown"

    return meta


def sid_card_text(meta: dict[str, Any], source_name: str) -> str:
    """Render parsed SID metadata as the searchable text of one document."""
    title = meta.get("title") or source_name
    lines = [
        f"# {title}",
        "",
        f"SID music file: `{source_name}`",
        "",
        "## Metadata",
        "",
        f"- Title: {meta.get('title') or '(untitled)'}",
        f"- Author: {meta.get('author') or '(unknown)'}",
        f"- Released: {meta.get('released') or '(unknown)'}",
        f"- Chip model: {meta.get('chip_model', 'Unknown')}",
        f"- Video standard: {meta.get('clock', 'Unknown')}",
        f"- Format: {meta.get('format')} v{meta.get('version')}",
        f"- Subtunes: {meta.get('songs')} (starts at {meta.get('start_song')})",
        f"- Load address: ${meta.get('load_address', 0):04X}",
        f"- Init address: ${meta.get('init_address', 0):04X}",
        f"- Play address: ${meta.get('play_address', 0):04X}",
    ]
    if meta.get("second_chip_model"):
        lines.append(
            f"- Second SID: {meta['second_chip_model']} at "
            f"${0xD000 + (meta.get('second_sid_address', 0) << 4):04X}"
        )
    if meta.get("third_chip_model"):
        lines.append(
            f"- Third SID: {meta['third_chip_model']} at "
            f"${0xD000 + (meta.get('third_sid_address', 0) << 4):04X}"
        )
    return "\n".join(lines) + "\n"


# A single HVSC member is a few KB; this cap stops a crafted archive from
# expanding into memory unbounded. It is deliberately generous - the largest
# real HVSC tunes are well under a megabyte.
MAX_MEMBER_BYTES = 4 * 1024 * 1024
MAX_MEMBERS = 5000


def sid_names_in_zip(zf: zipfile.ZipFile) -> list[str]:
    """Member names that look like SID tunes, in archive order."""
    return [
        info.filename for info in zf.infolist()
        if not info.is_dir() and info.filename.lower().endswith((".sid", ".psid", ".rsid"))
    ]


def parse_sid_zip(path: str) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Parse every SID member of a zip WITHOUT extracting anything to disk.

    Members are read through ZipFile.open straight into memory, so a hostile
    member name ("../../etc/passwd", an absolute path, a drive letter) has
    nothing to act on: this function never builds a filesystem path from a
    member name and never writes. That is a stronger guarantee than
    sanitising names during an extract, and it is why there is no sandbox
    directory here - there is nothing to sandbox.

    Returns (parsed, failures) where failures is [(member_name, reason)].
    """
    parsed: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    with zipfile.ZipFile(path) as zf:
        names = sid_names_in_zip(zf)
        if len(names) > MAX_MEMBERS:
            raise SidHeaderError(
                f"archive holds {len(names)} SID members, over the {MAX_MEMBERS} cap"
            )
        for name in names:
            info = zf.getinfo(name)
            if info.file_size > MAX_MEMBER_BYTES:
                failures.append((name, f"member declares {info.file_size} bytes, over the cap"))
                continue
            with zf.open(info) as fh:
                data = fh.read(MAX_MEMBER_BYTES + 1)
            if len(data) > MAX_MEMBER_BYTES:
                failures.append((name, "member expanded past the cap"))
                continue
            try:
                meta = parse_sid_header(data)
            except SidHeaderError as exc:
                failures.append((name, str(exc)))
                continue
            meta["member"] = name
            parsed.append(meta)
    return parsed, failures
