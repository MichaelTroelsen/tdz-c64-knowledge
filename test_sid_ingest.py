"""SID (PSID/RSID) ingestion tests.

The header layout asserted here comes from the published SID file format
specification shipped with HVSC (DOCUMENTS/SID_file_format.txt), not from a
sample file. Several of these tests exist specifically to fail a
plausible-but-wrong parser: a little-endian read, an invented chip model for
v1 files, or a name field that keeps its NUL padding all produce output that
looks reasonable until you compare it with the spec.
"""

import json
import struct
import zipfile

import pytest

from kb.ingest._sid import (
    DeepSidError,
    SidHeaderError,
    V1_HEADER_SIZE,
    V2_HEADER_SIZE,
    deepsid_card_text,
    deepsid_info_to_meta,
    deepsid_info_url,
    parse_deepsid_payload,
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


# --- DeepSID JSON API -------------------------------------------------------
#
# The reply below is shaped after DeepSID's php/info.php, which echoes
# {"status": "ok", "info": {...}} with these keys. No test here touches the
# network: the point of the endpoint's contract is that it can be pinned.

# The real collection_path format, established against the live endpoint: the
# collection folder carries a leading underscore and there is NO leading slash.
# The obvious-looking "/High Voltage SID Collection/..." is a MISS, and a miss
# is not an error (see test_a_path_that_matches_no_tune_is_refused), so getting
# this wrong yields an empty card rather than a failure.
DEEPSID_FULLNAME = "_High Voltage SID Collection/MUSICIANS/H/Hubbard_Rob/Commando.sid"

DEEPSID_INFO = {
    "player": "Rob_Hubbard",
    "lengths": "3:30 1:59",
    "type": "PSID",
    "version": "2",
    "playertype": "Rob Hubbard",
    "playercompat": "R64",
    "clockspeed": "PAL",
    "sidmodel": "MOS6581",
    "dataoffset": 124,
    "datasize": 3021,
    # PDO hands numbers back as strings; both shapes appear here on purpose.
    "loadaddr": "4096",
    "initaddr": 4099,
    "playaddr": 4102,
    "subtunes": "2",
    "startsubtune": "1",
    "name": "Commando",
    "author": "Rob Hubbard",
    "released": "1985 Elite",
    "hash": "d41d8cd98f00b204e9800998ecf8427e",
    # The live endpoint returns STIL with literal <br /> in it.
    "stil": "COMMENT: I went down to their<br />office and started working.",
}


def _ok(info=None):
    return json.dumps({"status": "ok", "info": info if info is not None else DEEPSID_INFO})


def test_info_url_percent_encodes_the_collection_path():
    """fullname holds spaces and slashes, so pasting it in unencoded breaks it."""
    url = deepsid_info_url(DEEPSID_FULLNAME)
    assert url.startswith("https://deepsid.chordian.net/php/info.php?")
    assert " " not in url
    assert "High+Voltage+SID+Collection" in url or "High%20Voltage%20SID%20Collection" in url
    assert "Commando.sid" in url.replace("%2F", "/").replace("+", " ")


def test_info_url_rejects_an_empty_fullname():
    with pytest.raises(DeepSidError):
        deepsid_info_url("   ")


def test_direct_access_refusal_names_the_missing_header():
    """php/info.php dies with a plain sentence, not JSON, when the header is absent.

    A bare json.loads reports a decode error about byte offsets, which sends
    the reader looking at the payload instead of at the request. This is the
    single likeliest way a first integration fails, so the message has to name
    the header.
    """
    with pytest.raises(DeepSidError) as exc:
        parse_deepsid_payload("Direct access not permitted.")
    assert "X-Requested-With" in str(exc.value)


def test_error_status_is_caught_even_though_it_arrives_as_http_200():
    """info.php reports failure in the body, so raise_for_status never fires."""
    body = json.dumps({"status": "error", "message": "Unknown file."})
    with pytest.raises(DeepSidError) as exc:
        parse_deepsid_payload(body)
    assert "Unknown file." in str(exc.value)


@pytest.mark.parametrize("body", ["", "   ", "<html>502</html>", "[1, 2, 3]",
                                  '{"status": "ok"}'])
def test_unusable_replies_raise_rather_than_yielding_an_empty_card(body):
    with pytest.raises(DeepSidError):
        parse_deepsid_payload(body)


def test_payload_returns_the_info_object_on_success():
    assert parse_deepsid_payload(_ok())["name"] == "Commando"


def test_numeric_fields_survive_arriving_as_strings():
    """loadaddr comes back as "4096" from PDO; the card formats it with :04X.

    Left as a string that reaches sid_card_text, this raises rather than
    rendering - so the coercion is load-bearing, not defensive decoration.
    """
    meta = deepsid_info_to_meta(DEEPSID_INFO)
    assert meta["load_address"] == 4096
    assert meta["init_address"] == 4099
    assert meta["songs"] == 2
    assert isinstance(meta["start_song"], int)


def test_missing_fields_become_unknown_rather_than_an_invented_default():
    """The local parser refuses to guess a chip model; so must this one."""
    meta = deepsid_info_to_meta({"name": "Sparse"})
    assert meta["chip_model"] == "Unknown"
    assert meta["clock"] == "Unknown"
    assert meta["load_address"] == 0


def test_card_carries_both_the_shared_fields_and_the_deepsid_only_ones():
    """The API is worth calling only for what the 124-byte header cannot hold."""
    text = deepsid_card_text(DEEPSID_INFO, DEEPSID_FULLNAME)
    for shared in ("Commando", "Rob Hubbard", "1985 Elite", "MOS6581", "PAL"):
        assert shared in text, f"{shared!r} missing from the DeepSID card"
    for api_only in ("3:30 1:59", "R64", "d41d8cd98f00b204e9800998ecf8427e"):
        assert api_only in text, f"{api_only!r} missing - the header cannot supply it"
    assert "office and started working." in text
    assert DEEPSID_FULLNAME in text


def test_a_path_that_matches_no_tune_is_refused(kb, monkeypatch):
    """A miss is not an error: status is ok and info holds only a PHP default.

    Confirmed against the live endpoint - a wrong path returns exactly
    {"sidmodel": "MOS6581"}, so rendering it produces a card that names a chip
    model for a tune that does not exist. That is the invented-default failure
    the local parser refuses to commit, arriving over the network instead.
    """
    miss = json.dumps({"status": "ok", "info": {"sidmodel": "MOS6581"}})
    monkeypatch.setattr("kb.ingest._extraction.http_get_polite",
                        lambda url, **kw: _FakeResponse(miss))
    monkeypatch.setattr("util.robots_allows", lambda url: True)

    with pytest.raises(DeepSidError) as exc:
        kb._extract_deepsid_metadata("/wrong/format/Commando.sid")
    assert "matched no tune" in str(exc.value)
    # the message has to teach the format, since the path IS the failure
    assert "_High Voltage SID Collection" in str(exc.value)


def test_stil_html_line_breaks_become_real_lines():
    """<br /> left in place would be indexed as part of the adjoining words."""
    text = deepsid_card_text(DEEPSID_INFO, DEEPSID_FULLNAME)
    assert "<br" not in text
    assert "their\noffice" in text


def test_card_renders_the_addresses_as_hex_like_the_local_card():
    assert "$1000" in deepsid_card_text(DEEPSID_INFO, DEEPSID_FULLNAME)


def test_card_omits_absent_optional_sections_instead_of_printing_blanks():
    text = deepsid_card_text({"name": "Bare", "type": "PSID", "version": "2"},
                             "/x/Bare.sid")
    assert "STIL entry" not in text
    assert "Song lengths" not in text
    assert "Bare" in text


class _FakeResponse:
    def __init__(self, body, status_code=200):
        self.text = body
        self.status_code = status_code


def test_mixin_fetch_sends_the_xhr_header_without_dropping_the_user_agent(kb, monkeypatch):
    """http_get_polite does kwargs.setdefault('headers', ...).

    So passing a bare {'X-Requested-With': ...} REPLACES the identifying
    User-Agent that the politeness work exists to send, silently and without
    failing any other assertion here. This pins both headers.
    """
    seen = {}

    def fake_get(url, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs.get("headers", {})
        return _FakeResponse(_ok())

    monkeypatch.setattr("kb.ingest._extraction.http_get_polite", fake_get)
    monkeypatch.setattr("util.robots_allows", lambda url: True)

    text = kb._extract_deepsid_metadata(DEEPSID_FULLNAME)

    assert seen["headers"].get("X-Requested-With") == "XMLHttpRequest"
    assert "tdz-c64-knowledge" in seen["headers"].get("User-Agent", "")
    assert "info.php" in seen["url"]
    assert "Commando" in text and "3:30 1:59" in text


def test_mixin_refuses_a_non_200_before_blaming_the_body(kb, monkeypatch):
    monkeypatch.setattr("kb.ingest._extraction.http_get_polite",
                        lambda url, **kw: _FakeResponse("<html>not found</html>", 404))
    monkeypatch.setattr("util.robots_allows", lambda url: True)

    with pytest.raises(DeepSidError) as exc:
        kb._extract_deepsid_metadata(DEEPSID_FULLNAME)
    assert "404" in str(exc.value)


def test_mixin_honours_robots_txt(kb, monkeypatch):
    monkeypatch.setattr("util.robots_allows", lambda url: False)
    monkeypatch.setattr("kb.ingest._extraction.http_get_polite",
                        lambda url, **kw: pytest.fail("fetched despite robots.txt"))

    with pytest.raises(DeepSidError) as exc:
        kb._extract_deepsid_metadata(DEEPSID_FULLNAME)
    assert "robots" in str(exc.value).lower()


def test_expected_methods_guard_lists_the_new_mixin_method():
    """The guard is extended, not weakened - it must still name every method."""
    from kb.ingest import _EXPECTED_METHODS
    from kb.ingest._extraction import _ExtractionMixin

    assert "_extract_deepsid_metadata" in _EXPECTED_METHODS
    assert hasattr(_ExtractionMixin, "_extract_deepsid_metadata")
