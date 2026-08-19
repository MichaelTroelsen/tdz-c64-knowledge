"""SID (PSID/RSID) ingestion tests.

The header layout asserted here comes from the published SID file format
specification shipped with HVSC (DOCUMENTS/SID_file_format.txt), not from a
sample file. Several of these tests exist specifically to fail a
plausible-but-wrong parser: a little-endian read, an invented chip model for
v1 files, or a name field that keeps its NUL padding all produce output that
looks reasonable until you compare it with the spec.
"""

import struct
import zipfile

import pytest

from kb.ingest._sid import (
    SidHeaderError,
    V1_HEADER_SIZE,
    V2_HEADER_SIZE,
    parse_sid_header,
    parse_sid_zip,
    sid_card_text,
)


def build_sid(version=2, magic=b"PSID", title=b"Commando", author=b"Rob Hubbard",
              released=b"1985", flags=0, songs=1, start_song=1, load_address=0x1000,
              data_offset=None, trailer=b"\x00\x00"):
    """Build a header byte-for-byte at the spec's offsets."""
    size = V1_HEADER_SIZE if version == 1 else V2_HEADER_SIZE
    if data_offset is None:
        data_offset = size
    h = bytearray(size)
    h[0x00:0x04] = magic
    struct.pack_into(">H", h, 0x04, version)
    struct.pack_into(">H", h, 0x06, data_offset)
    struct.pack_into(">H", h, 0x08, load_address)
    struct.pack_into(">H", h, 0x0A, 0x1003)      # initAddress
    struct.pack_into(">H", h, 0x0C, 0x1006)      # playAddress
    struct.pack_into(">H", h, 0x0E, songs)
    struct.pack_into(">H", h, 0x10, start_song)
    struct.pack_into(">I", h, 0x12, 0)           # speed
    h[0x16:0x16 + len(title)] = title
    h[0x36:0x36 + len(author)] = author
    h[0x56:0x56 + len(released)] = released
    if version >= 2:
        struct.pack_into(">H", h, 0x76, flags)
    return bytes(h) + trailer


def test_fields_land_at_the_spec_offsets():
    meta = parse_sid_header(build_sid())
    assert meta["format"] == "PSID"
    assert meta["version"] == 2
    assert meta["title"] == "Commando"
    assert meta["author"] == "Rob Hubbard"
    assert meta["released"] == "1985"
    assert meta["init_address"] == 0x1003
    assert meta["play_address"] == 0x1006


def test_multibyte_fields_are_big_endian():
    """The spec's LEGEND defines WORD as big endian.

    songs=0x0100 is 256 read big-endian and 1 read little-endian, so a parser
    with the byte order wrong returns a believable number instead of failing.
    """
    meta = parse_sid_header(build_sid(songs=0x0100))
    assert meta["songs"] == 256


@pytest.mark.parametrize("bits,expected", [
    (0b00, "Unknown"),
    (0b01, "MOS6581"),
    (0b10, "MOS8580"),
    (0b11, "MOS6581 and MOS8580"),
])
def test_chip_model_from_flags_bits_4_and_5(bits, expected):
    meta = parse_sid_header(build_sid(flags=bits << 4))
    assert meta["chip_model"] == expected


@pytest.mark.parametrize("bits,expected", [
    (0b00, "Unknown"),
    (0b01, "PAL"),
    (0b10, "NTSC"),
    (0b11, "PAL and NTSC"),
])
def test_video_standard_from_flags_bits_2_and_3(bits, expected):
    meta = parse_sid_header(build_sid(flags=bits << 2))
    assert meta["clock"] == expected


def test_chip_model_and_clock_are_independent_bitfields():
    """Setting one must not bleed into the other."""
    meta = parse_sid_header(build_sid(flags=(0b10 << 4) | (0b01 << 2)))
    assert meta["chip_model"] == "MOS8580"
    assert meta["clock"] == "PAL"


def test_v1_has_no_flags_word_so_chip_model_is_unknown():
    """A v1 header is 118 bytes and stops before flags.

    Reporting 6581 here because most v1 tunes were 6581 would be inventing a
    fact the file does not carry.
    """
    data = build_sid(version=1)
    assert len(data) == V1_HEADER_SIZE + 2
    meta = parse_sid_header(data)
    assert meta["chip_model"] == "Unknown"
    assert meta["clock"] == "Unknown"
    assert "flags" not in meta


def test_load_address_zero_is_read_little_endian_from_the_data():
    """The one little-endian value in an otherwise big-endian format."""
    meta = parse_sid_header(build_sid(load_address=0, trailer=b"\x00\x10"))
    assert meta["load_address"] == 0x1000
    assert meta["load_address_from_data"] is True


def test_strings_stop_at_the_nul_and_drop_padding():
    meta = parse_sid_header(build_sid(title=b"Zoids\x00garbage-after-nul"))
    assert meta["title"] == "Zoids"


def test_strings_decode_as_windows_1252():
    """Spec: extended ASCII, Windows-1252 code page. 0xE9 is e-acute."""
    meta = parse_sid_header(build_sid(author=b"Bj\xf8rn caf\xe9"))
    assert meta["author"] == "Bjørn café"


def test_rsid_magic_is_accepted():
    assert parse_sid_header(build_sid(magic=b"RSID"))["format"] == "RSID"


@pytest.mark.parametrize("data,fragment", [
    (build_sid(magic=b"MP3 "), "bad magic"),
    (build_sid()[:50], "shorter than"),
    (b"", "shorter than"),
])
def test_malformed_files_raise_rather_than_half_parse(data, fragment):
    with pytest.raises(SidHeaderError) as exc:
        parse_sid_header(data)
    assert fragment in str(exc.value)


def test_unsupported_version_is_rejected():
    bad = bytearray(build_sid())
    struct.pack_into(">H", bad, 0x04, 5)
    with pytest.raises(SidHeaderError) as exc:
        parse_sid_header(bytes(bad))
    assert "version 5" in str(exc.value)


def test_card_text_carries_the_four_fields_the_task_requires():
    text = sid_card_text(parse_sid_header(build_sid(flags=0b01 << 4)), "Commando.sid")
    assert "Commando" in text
    assert "Rob Hubbard" in text
    assert "1985" in text
    assert "MOS6581" in text


def _write_zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return path


def test_zip_parses_every_sid_member(tmp_path):
    z = _write_zip(tmp_path / "hvsc.zip", {
        "MUSICIANS/H/Hubbard_Rob/Commando.sid": build_sid(title=b"Commando"),
        "MUSICIANS/G/Galway_Martin/Rambo.sid": build_sid(title=b"Rambo"),
        "DOCUMENTS/readme.txt": b"not a tune",
    })
    parsed, failures = parse_sid_zip(str(z))
    assert [m["title"] for m in parsed] == ["Commando", "Rambo"]
    assert failures == []


def test_zip_reports_unparseable_members_instead_of_dropping_them(tmp_path):
    z = _write_zip(tmp_path / "mixed.zip", {
        "good.sid": build_sid(title=b"Good"),
        "broken.sid": b"MP3 not really a sid file at all",
    })
    parsed, failures = parse_sid_zip(str(z))
    assert [m["title"] for m in parsed] == ["Good"]
    assert len(failures) == 1
    assert failures[0][0] == "broken.sid"


def test_hostile_member_names_write_nothing_outside_the_archive(tmp_path):
    """Zip-slip has nothing to act on because nothing is ever extracted.

    The members below are the classic escapes - parent traversal, an absolute
    path and a Windows drive path. The assertion is not that the names were
    sanitised, it is that no file appeared anywhere: parse_sid_zip reads
    members into memory and never builds a path from a member name.
    """
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    z = _write_zip(sandbox / "evil.zip", {
        "../../outside/escaped.sid": build_sid(title=b"Escaped"),
        "/tmp/absolute.sid": build_sid(title=b"Absolute"),
        "normal.sid": build_sid(title=b"Normal"),
    })

    before = {p.name for p in outside.iterdir()}
    parsed, _ = parse_sid_zip(str(z))

    # the members were still READ - this is not passing by ignoring them
    assert {m["title"] for m in parsed} == {"Escaped", "Absolute", "Normal"}
    # ...but nothing was written anywhere
    assert {p.name for p in outside.iterdir()} == before == set()
    assert list(sandbox.iterdir()) == [sandbox / "evil.zip"]


@pytest.fixture
def kb(tmp_path, monkeypatch):
    """A KnowledgeBase on an isolated data dir, never the live one."""
    monkeypatch.setenv("TDZ_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("USE_BM25", "0")
    from server import KnowledgeBase
    instance = KnowledgeBase(str(tmp_path))
    yield instance
    instance.close()


def test_sid_file_ingests_into_a_searchable_document(kb, tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir(exist_ok=True)
    sid = uploads / "Commando.sid"
    sid.write_bytes(build_sid(title=b"Commando", author=b"Rob Hubbard",
                              released=b"1985", flags=0b01 << 4))

    doc = kb.add_document(str(sid))
    assert doc.file_type == "sid"

    text = "\n".join(c.content for c in kb._get_chunks_db(doc.doc_id))
    for field in ("Commando", "Rob Hubbard", "1985", "MOS6581"):
        assert field in text, f"{field!r} missing from the ingested card"

    hits = kb.search("Hubbard", max_results=10)
    assert any((h["doc_id"] if isinstance(h, dict) else h.doc_id) == doc.doc_id for h in hits), hits


def test_zip_archive_ingests_its_members(kb, tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir(exist_ok=True)
    z = _write_zip(uploads / "hvsc-sample.zip", {
        "Hubbard_Rob/Commando.sid": build_sid(title=b"Commando", author=b"Rob Hubbard"),
        "Galway_Martin/Rambo.sid": build_sid(title=b"Rambo", author=b"Martin Galway"),
    })

    doc = kb.add_document(str(z))
    assert doc.file_type == "sid-archive"
    assert doc.total_pages == 2

    text = "\n".join(c.content for c in kb._get_chunks_db(doc.doc_id))
    for field in ("Commando", "Rob Hubbard", "Rambo", "Martin Galway"):
        assert field in text, f"{field!r} missing from the archive card"


def test_zip_with_no_sid_members_is_refused(kb, tmp_path):
    from models import UnsupportedFileTypeError

    uploads = tmp_path / "uploads"
    uploads.mkdir(exist_ok=True)
    z = _write_zip(uploads / "docs.zip", {"readme.txt": b"no tunes here"})
    with pytest.raises((UnsupportedFileTypeError, Exception)) as exc:
        kb.add_document(str(z))
    assert "no readable SID" in str(exc.value)
