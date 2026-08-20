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


# --- DeepSID JSON API -------------------------------------------------------
#
# DeepSID (https://deepsid.chordian.net) is a single-page app: the HTML served
# at "/" is navigation chrome, and every tune's metadata arrives over a JSON
# endpoint instead. Scraping the page as markdown therefore captures menus and
# no music data, which is why this goes through the API.
#
# The endpoint and its contract are taken from DeepSID's own source
# (github.com/Chordian/deepsid, php/info.php), not inferred from traffic:
#
#   - it reads exactly one GET parameter, `fullname`, the tune's path within
#     the collection, and looks it up as `files.collection_path`;
#   - it answers {"status": "ok", "info": {...}} or
#     {"status": "error", "message": ...};
#   - it opens with
#         if (!isset($_SERVER['HTTP_X_REQUESTED_WITH']) ||
#             $_SERVER['HTTP_X_REQUESTED_WITH'] != 'XMLHttpRequest')
#             die("Direct access not permitted.");
#     so without that header the body is a plain sentence rather than JSON.
#
# Both of those are silent traps. The error reply carries HTTP 200, so
# raise_for_status() never fires on it, and the refusal is not JSON at all, so
# a bare json.loads raises a decode error that names a column offset instead of
# the missing header. parse_deepsid_payload() below turns each into a sentence
# that names its own cause.

DEEPSID_BASE_URL = "https://deepsid.chordian.net"

# php/info.php's own words, matched to explain the failure rather than guessed.
_DEEPSID_REFUSAL = "Direct access not permitted"


class DeepSidError(RuntimeError):
    """Raised when DeepSID's reply cannot be read as tune metadata."""


def deepsid_info_url(fullname: str, base_url: str = DEEPSID_BASE_URL) -> str:
    """URL of the info endpoint for one tune path.

    `fullname` is a collection path such as
    "/High Voltage SID Collection/MUSICIANS/H/Hubbard_Rob/Commando.sid" - it
    holds spaces and slashes, so it is percent-encoded rather than pasted in.
    """
    from urllib.parse import urlencode

    if not fullname or not fullname.strip():
        raise DeepSidError("fullname is required: it is the tune's path within the collection")
    return f"{base_url.rstrip('/')}/php/info.php?" + urlencode({"fullname": fullname})


def parse_deepsid_payload(body: str) -> dict[str, Any]:
    """Validate one info.php reply and return its `info` object.

    Every rejection below names the cause in the message, because all three
    failures otherwise surface as something misleading: a decode error about
    byte offsets, a KeyError, or a card rendered from an empty dict.
    """
    import json

    text = (body or "").strip()
    if not text:
        raise DeepSidError("DeepSID returned an empty body")

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        if _DEEPSID_REFUSAL.lower() in text.lower():
            raise DeepSidError(
                "DeepSID refused the request as direct access: info.php requires the "
                "header 'X-Requested-With: XMLHttpRequest'"
            ) from exc
        raise DeepSidError(f"DeepSID returned a non-JSON body: {text[:120]!r}") from exc

    if not isinstance(payload, dict):
        raise DeepSidError(f"DeepSID returned {type(payload).__name__}, expected a JSON object")

    status = payload.get("status")
    if status != "ok":
        # Carried at HTTP 200, so this is the only place it can be caught.
        message = payload.get("message") or "(no message)"
        raise DeepSidError(f"DeepSID reported status {status!r}: {message}")

    info = payload.get("info")
    if not isinstance(info, dict):
        raise DeepSidError("DeepSID reply has status ok but no 'info' object")

    # A path that matches no row is NOT reported as an error: info.php runs
    # "SELECT * FROM files WHERE collection_path = :collection_path LIMIT 1",
    # gets nothing, and still answers status ok with an $info holding only its
    # own SID-model default. Confirmed live: a miss returns {"sidmodel":
    # "MOS6581"} and a hit returns 20 keys. Rendered as-is that is a card
    # asserting a chip model for a tune that was never found, which is the
    # invented-default failure parse_sid_header already refuses to commit.
    # `name` is the tune's identity and is present on every real row, so its
    # absence is the miss. A real tune with a genuinely empty name would be
    # refused too; that is the safer direction to be wrong in.
    if not str(info.get("name") or "").strip():
        raise DeepSidError(
            "DeepSID matched no tune for that path (it answers status ok with an "
            "empty record rather than an error). collection_path starts with the "
            "collection folder and no leading slash, e.g. "
            "'_High Voltage SID Collection/MUSICIANS/H/Hubbard_Rob/Commando.sid'"
        )
    return info


def _as_int(value: Any) -> int:
    """Coerce one address/count field to int.

    The live endpoint was observed returning these as ints, but the same
    columns arrive as strings from a PDO connection with emulated prepares on
    - and sid_card_text formats them with :04X, which raises on a str. Both
    shapes are therefore accepted rather than assuming the one seen today.
    """
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip() or 0, 0)
    except (TypeError, ValueError):
        return 0


def deepsid_info_to_meta(info: dict[str, Any]) -> dict[str, Any]:
    """Map DeepSID's field names onto the PSID header vocabulary.

    parse_sid_header already defines the names the card renderer reads
    (title/author/chip_model/clock/...), and info.php uses its own
    (name/sidmodel/clockspeed/...). Translating here means one card format
    serves both sources instead of two that drift apart.

    Absent fields become "Unknown" rather than a plausible default: the local
    parser already refuses to invent a chip model for v1 files, and inventing
    one for a sparse API row would be the same error by another route.
    """
    return {
        "title": (info.get("name") or "").strip(),
        "author": (info.get("author") or "").strip(),
        "released": (info.get("released") or "").strip(),
        "format": (info.get("type") or "").strip() or "Unknown",
        "version": (str(info.get("version") or "").strip() or "Unknown"),
        "chip_model": (info.get("sidmodel") or "").strip() or "Unknown",
        "clock": (info.get("clockspeed") or "").strip() or "Unknown",
        "songs": _as_int(info.get("subtunes")),
        "start_song": _as_int(info.get("startsubtune")),
        "load_address": _as_int(info.get("loadaddr")),
        "init_address": _as_int(info.get("initaddr")),
        "play_address": _as_int(info.get("playaddr")),
    }


def _stil_to_text(raw: Any) -> str:
    """Flatten the STIL entry's embedded HTML into plain lines.

    The API returns STIL with literal <br /> tags in it (confirmed against the
    live record for Commando.sid). Left in place they end up indexed as part
    of the words around them, so a search for the last word of a line would
    have to match "office<br />and". Only the line break appears in practice;
    other markup is left alone rather than half-stripped by a general
    tag-removal pass that would also eat legitimate angle brackets.
    """
    import re

    text = str(raw or "")
    if not text.strip():
        return ""
    return re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE).strip()


def deepsid_card_text(info: dict[str, Any], fullname: str,
                      base_url: str = DEEPSID_BASE_URL) -> str:
    """Render one DeepSID info reply as the searchable text of a document.

    The shared fields go through sid_card_text so a tune reads the same
    whether it came from a local .sid file or from the API. Appended after it
    is the part that justifies the network call at all: the player routine,
    per-subtune lengths and the STIL entry are curated by HVSC and exist
    nowhere in the file's own 124-byte header.
    """
    meta = deepsid_info_to_meta(info)
    source_name = fullname.rsplit("/", 1)[-1] or fullname
    parts = [sid_card_text(meta, source_name), ""]

    parts.append("## DeepSID")
    parts.append("")
    parts.append(f"- Collection path: `{fullname}`")
    parts.append(f"- Source: {deepsid_info_url(fullname, base_url)}")
    for label, key in (("Player", "player"), ("Player type", "playertype"),
                       ("Player compatibility", "playercompat"),
                       ("Song lengths", "lengths"), ("MD5 hash", "hash")):
        value = str(info.get(key) or "").strip()
        if value:
            parts.append(f"- {label}: {value}")

    stil = _stil_to_text(info.get("stil"))
    if stil:
        # STIL is prose (composer notes, cover origins, trivia) and the most
        # searchable text on the card, so it gets its own section rather than
        # being flattened into a bullet.
        parts.extend(["", "## STIL entry", "", stil])

    return "\n".join(parts) + "\n"
