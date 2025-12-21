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
    """Extract all MCP tool names from server.py."""
    server_path = Path("server.py")
    tools = set()

    if not server_path.exists():
        print(f"{Colors.RED}Error: server.py not found{Colors.END}")
        return tools

    content = server_path.read_text(encoding='utf-8')

    # Find all Tool definitions with name parameter
    pattern = r'Tool\s*\(\s*name\s*=\s*"([^"]+)"'
    matches = re.findall(pattern, content)

    for match in matches:
        tools.add(match)

    return tools


def extract_documented_tools_from_readme() -> Set[str]:
    """Extract all documented tool names from README.md."""
    readme_path = Path("README.md")
    tools = set()

    if not readme_path.exists():
        print(f"{Colors.RED}Error: README.md not found{Colors.END}")
        return tools

    content = readme_path.read_text(encoding='utf-8')

    # Find tool headers (### or #### followed by tool_name)
    # Matches both ### tool_name and #### tool_name
    pattern = r'^#{3,4}\s+([a-z_]+)\s*$'
    matches = re.findall(pattern, content, re.MULTILINE)

    for match in matches:
        tools.add(match)

    return tools


def extract_features_from_changelog() -> Dict[str, List[str]]:
    """Extract features from CHANGELOG.md by version."""
    changelog_path = Path("CHANGELOG.md")
    features = {}

    if not changelog_path.exists():
        print(f"{Colors.RED}Error: CHANGELOG.md not found{Colors.END}")
        return features

    content = changelog_path.read_text(encoding='utf-8')

    # Extract version sections
    version_pattern = r'##\s+\[?(\d+\.\d+\.\d+)\]?'
    versions = re.findall(version_pattern, content)

    for version in versions[:5]:  # Check last 5 versions
        # Extract feature bullets for this version
        version_section_pattern = rf'##\s+\[?{re.escape(version)}\]?.*?(?=##|\Z)'
        section_match = re.search(version_section_pattern, content, re.DOTALL)

        if section_match:
            section = section_match.group(0)
            # Find bullet points
            bullet_pattern = r'^\s*[-*]\s+\*\*([^*]+)\*\*'
            feature_names = re.findall(bullet_pattern, section, re.MULTILINE)
            features[version] = feature_names

    return features


def check_version_consistency() -> Tuple[bool, List[str]]:
    """Check version numbers are consistent across files."""
    files_to_check = {
        'version.py': r'__version__\s*=\s*"([^"]+)"',
        'README.md': r'\[!\[Version\]\([^)]*version-([0-9.]+)',
        'QUICKSTART.md': r'\*\*Version:\*\*\s+([0-9.]+)'
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

    print(f"   Tools in server.py: {len(server_tools)}")
    print(f"   Tools in README.md: {len(readme_tools)}")

    missing_in_readme = server_tools - readme_tools
    extra_in_readme = readme_tools - server_tools

    if missing_in_readme:
        print(f"   {Colors.RED}X Tools not documented in README:{Colors.END}")
        for tool in sorted(missing_in_readme):
            print(f"     - {tool}")
        all_passed = False

    if extra_in_readme:
        print(f"   {Colors.YELLOW}! Tools in README but not in server.py:{Colors.END}")
        for tool in sorted(extra_in_readme):
            print(f"     - {tool}")

    if not missing_in_readme and not extra_in_readme:
        print(f"   {Colors.GREEN}+ All MCP tools properly documented{Colors.END}")

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
