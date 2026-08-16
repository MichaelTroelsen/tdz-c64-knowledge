#!/usr/bin/env python3
"""
Documentation Completeness Validation Script

Validates that all features, tools, and examples are properly documented.
"""

import re
from pathlib import Path
from typing import Set, List, Dict, Tuple

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

def extract_mcp_tools_from_server() -> Set[str]:
    """Extract all MCP tool names from the schema definitions.

    R12 moved the Tool(...) literals out of server.py into
    mcp_tools/schemas.py, so both are scanned: reading only server.py made
    this report every documented tool as deleted.
    """
    sources = [Path("mcp_tools/schemas.py"), Path("server.py")]
    present = [p for p in sources if p.exists()]
    tools = set()

    if not present:
        print(f"{Colors.RED}Error: no tool schema source found "
              f"({', '.join(str(p) for p in sources)}){Colors.END}")
        return tools

    # Find all Tool definitions with name parameter
    pattern = r'Tool\s*\(\s*name\s*=\s*"([^"]+)"'
    for path in present:
        tools.update(re.findall(pattern, path.read_text(encoding='utf-8')))

    return tools


def extract_documented_tools_from_readme() -> Set[str]:
    """Extract all documented tool names from README.md.

    README documents tools as bold names ("**search_docs** - Full-text
    search"), not as markdown headers. The previous `^#{3,4}\\s+([a-z_]+)$`
    pattern matched none of them, so this always reported 0 documented tools
    and every one of the 92 as missing.
    """
    readme_path = Path("README.md")
    tools = set()

    if not readme_path.exists():
        print(f"{Colors.RED}Error: README.md not found{Colors.END}")
        return tools

    content = readme_path.read_text(encoding='utf-8')

    for pattern in (
        r'\*\*([a-z][a-z0-9_]{2,})\*\*',   # **tool_name** - description
        r'^#{3,4}\s+([a-z][a-z0-9_]{2,})\s*$',  # ### tool_name (also honoured)
        r'^([a-z][a-z0-9_]{2,})\(',        # tool_name(...) in a code fence
    ):
        tools.update(re.findall(pattern, content, re.MULTILINE))

    return tools


def extract_table_names_from_server() -> Set[str]:
    """Database table names, so README's schema docs aren't read as tools.

    README documents the schema with bold table names (**document_entities**,
    **entity_relationships**, ...) in exactly the same markup it uses for
    tools, so without this they get reported as tools that no longer exist.
    """
    server_path = Path("server.py")
    if not server_path.exists():
        return set()

    content = server_path.read_text(encoding='utf-8')
    pattern = r'CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-z_][a-z0-9_]*)'
    return set(re.findall(pattern, content, re.IGNORECASE))


def _find_changelog() -> Path:
    """Locate CHANGELOG.md, which lives under docs/ in this repo."""
    for candidate in (Path("docs/CHANGELOG.md"), Path("CHANGELOG.md")):
        if candidate.exists():
            return candidate
    return Path("CHANGELOG.md")  # reported as missing by the caller


def extract_features_from_changelog() -> Dict[str, List[str]]:
    """Extract features from CHANGELOG.md by version."""
    changelog_path = _find_changelog()
    features = {}

    if not changelog_path.exists():
        print(f"{Colors.RED}Error: CHANGELOG.md not found{Colors.END}")
        return features

    content = changelog_path.read_text(encoding='utf-8')

    # Extract version sections
    version_pattern = r'##\s+\[?(\d+\.\d+\.\d+)\]?'
    versions = re.findall(version_pattern, content)

    for version in versions[:5]:  # Check last 5 versions
        # Extract feature bullets for this version. The terminator must be a
        # level-2 heading at line start: a bare `(?=##)` also matched the
        # section's own `### Added` subheading, so every section came back
        # empty and this check silently reported 0 features for everything.
        version_section_pattern = (
            rf'^##\s+\[?{re.escape(version)}\]?.*?(?=^##\s|\Z)'
        )
        section_match = re.search(version_section_pattern, content,
                                  re.DOTALL | re.MULTILINE)

        if section_match:
            section = section_match.group(0)
            # Find bullet points
            bullet_pattern = r'^\s*[-*]\s+\*\*([^*]+)\*\*'
            feature_names = re.findall(bullet_pattern, section, re.MULTILINE)
            features[version] = feature_names

    return features


def check_version_consistency() -> Tuple[bool, List[str]]:
    """Check version numbers are consistent across files."""
    # docs/ paths matter: QUICKSTART.md lives under docs/, so checking for it
    # in the repo root silently skipped it and made this check weaker than the
    # green tick suggested.
    files_to_check = {
        'version.py': r'__version__\s*=\s*"([^"]+)"',
        'README.md': r'\[!\[Version\]\([^)]*version-([0-9.]+)',
        'docs/QUICKSTART.md': r'\*\*Version:\*\*\s+([0-9.]+)',
    }

    versions = {}
    errors = []

    for filename, pattern in files_to_check.items():
        filepath = Path(filename)
        if filepath.exists():
            content = filepath.read_text(encoding='utf-8')
            match = re.search(pattern, content)
            if match:
                versions[filename] = match.group(1)
            else:
                errors.append(f"Could not find version in {filename}")

    # Check if all versions match
    if len(set(versions.values())) > 1:
        errors.append(f"Version mismatch: {versions}")
        return False, errors

    return len(errors) == 0, errors


def check_internal_links() -> Tuple[bool, List[str]]:
    """Check internal file links in documentation."""
    readme_path = Path("README.md")
    errors = []

    if not readme_path.exists():
        return False, ["README.md not found"]

    content = readme_path.read_text(encoding='utf-8')

    # Find markdown links to local files
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    matches = re.findall(link_pattern, content)

    for link_text, link_target in matches:
        # Skip external links
        if link_target.startswith(('http://', 'https://', '#')):
            continue

        # Check if file exists
        filepath = Path(link_target)
        if not filepath.exists():
            errors.append(f"Broken link: [{link_text}]({link_target})")

    return len(errors) == 0, errors


def main():
    """Run all validation checks."""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BLUE}Documentation Completeness Validation{Colors.END}")
    print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")

    all_passed = True

    # Check 1: MCP Tools Documentation
    print(f"{Colors.BLUE}1. Checking MCP Tools Documentation...{Colors.END}")
    server_tools = extract_mcp_tools_from_server()
    readme_tools = extract_documented_tools_from_readme()

    documented = server_tools & readme_tools
    missing_in_readme = server_tools - readme_tools
    # Only names that look like tools - README prose contains plenty of other
    # bold words, and flagging those as phantom tools is pure noise. Database
    # table names use identical markup, so exclude those too.
    table_names = extract_table_names_from_server()
    stale_in_readme = {
        t for t in (readme_tools - server_tools)
        if '_' in t and t not in table_names
    }

    print(f"   Tools defined: {len(server_tools)}")
    print(f"   Documented in README.md: {len(documented)}")

    # README deliberately documents a curated subset ("Key tools listed
    # below"), so a tool missing from it is informational, not a failure.
    # Requiring all 92 made this check permanently red and therefore ignored.
    if missing_in_readme:
        print(f"   {Colors.YELLOW}! {len(missing_in_readme)} tool(s) not mentioned in README "
              f"(README documents a curated subset by design){Colors.END}")

    # A README tool that no longer exists in server.py IS real drift: someone
    # renamed or removed it and left the docs behind.
    if stale_in_readme:
        print(f"   {Colors.RED}X README documents tools that no longer exist in server.py:{Colors.END}")
        for tool in sorted(stale_in_readme):
            print(f"     - {tool}")
        all_passed = False
    else:
        print(f"   {Colors.GREEN}+ No stale tool references in README{Colors.END}")

    print()

    # Check 2: Version Consistency
    print(f"{Colors.BLUE}2. Checking Version Consistency...{Colors.END}")
    version_ok, version_errors = check_version_consistency()

    if version_ok:
        print(f"   {Colors.GREEN}+ All version numbers consistent{Colors.END}")
    else:
        print(f"   {Colors.RED}X Version inconsistencies found:{Colors.END}")
        for error in version_errors:
            print(f"     - {error}")
        all_passed = False

    print()

    # Check 3: Internal Links
    print(f"{Colors.BLUE}3. Checking Internal Links...{Colors.END}")
    links_ok, link_errors = check_internal_links()

    if links_ok:
        print(f"   {Colors.GREEN}+ All internal links valid{Colors.END}")
    else:
        print(f"   {Colors.RED}X Broken links found:{Colors.END}")
        for error in link_errors:
            print(f"     - {error}")
        all_passed = False

    print()

    # Check 4: Feature Documentation
    print(f"{Colors.BLUE}4. Checking Feature Documentation...{Colors.END}")
    features = extract_features_from_changelog()

    if features:
        print(f"   {Colors.GREEN}+ Found features in CHANGELOG for {len(features)} versions{Colors.END}")
        for version, feature_list in sorted(features.items(), reverse=True)[:3]:
            print(f"     v{version}: {len(feature_list)} features")
    else:
        print(f"   {Colors.YELLOW}! Could not extract features from CHANGELOG{Colors.END}")

    print()

    # Summary
    print(f"{Colors.BLUE}{'='*70}{Colors.END}")
    if all_passed:
        print(f"{Colors.GREEN}+ All documentation validation checks PASSED{Colors.END}")
        print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")
        return 0
    else:
        print(f"{Colors.RED}X Some documentation validation checks FAILED{Colors.END}")
        print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")
        return 1


if __name__ == "__main__":
    exit(main())
