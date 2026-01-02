#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup Claude Desktop MCP configuration for C64 Knowledge Base."""

import os
import json
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 70)
print("Claude Desktop MCP Configuration Setup")
print("=" * 70)

# Step 1: Find Claude Desktop config location
print("\n1. Locating Claude Desktop config file...")

if sys.platform == 'win32':
    appdata = os.getenv('APPDATA')
    if not appdata:
        print("✗ APPDATA environment variable not found")
        sys.exit(1)

    config_dir = Path(appdata) / "Claude"
    config_file = config_dir / "claude_desktop_config.json"
else:
    print("✗ This script is for Windows only")
    sys.exit(1)

print(f"   Config directory: {config_dir}")
print(f"   Config file: {config_file}")

# Step 2: Check if directory exists
if not config_dir.exists():
    print(f"\n⚠ Claude Desktop config directory not found!")
    print(f"   Creating directory: {config_dir}")
    config_dir.mkdir(parents=True, exist_ok=True)
    print("   ✓ Directory created")

# Step 3: Load existing config or create new
existing_config = {}
if config_file.exists():
    print(f"\n2. Found existing config file")
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            existing_config = json.load(f)
        print(f"   ✓ Loaded existing configuration")

        if 'mcpServers' in existing_config:
            print(f"   Existing MCP servers: {list(existing_config['mcpServers'].keys())}")
    except Exception as e:
        print(f"   ⚠ Error reading config: {e}")
        print(f"   Will create new configuration")
else:
    print(f"\n2. No existing config file found")
    print(f"   Will create new configuration")

# Step 4: Build C64 Knowledge Base MCP server config
print(f"\n3. Building MCP server configuration...")

# Get absolute paths
current_dir = Path(__file__).parent.absolute()
venv_python = current_dir / ".venv" / "Scripts" / "python.exe"
server_script = current_dir / "server.py"
poppler_path = current_dir / "poppler-25.12.0" / "Library" / "bin"
data_dir = Path.home() / ".tdz-c64-knowledge"

# Verify paths exist
if not venv_python.exists():
    print(f"   ✗ Python executable not found: {venv_python}")
    sys.exit(1)

if not server_script.exists():
    print(f"   ✗ Server script not found: {server_script}")
    sys.exit(1)

print(f"   ✓ Python: {venv_python}")
print(f"   ✓ Server: {server_script}")
print(f"   ✓ Poppler: {poppler_path}")

# Build the MCP server config
mcp_config = {
    "command": str(venv_python).replace('\\', '\\\\'),
    "args": [str(server_script).replace('\\', '\\\\')],
    "env": {
        "USE_FTS5": "1",
        "USE_OCR": "1",
        "POPPLER_PATH": str(poppler_path).replace('\\', '\\\\'),
        "TDZ_DATA_DIR": str(data_dir).replace('\\', '\\\\')
    }
}

# Step 5: Merge with existing config
if 'mcpServers' not in existing_config:
    existing_config['mcpServers'] = {}

# Check if c64-knowledge already exists
if 'c64-knowledge' in existing_config['mcpServers']:
    print(f"\n⚠ 'c64-knowledge' MCP server already configured")
    print(f"   Do you want to overwrite? (y/n): ", end='')

    # For automated setup, just overwrite
    print("yes (automated)")
    existing_config['mcpServers']['c64-knowledge'] = mcp_config
else:
    existing_config['mcpServers']['c64-knowledge'] = mcp_config
    print(f"   ✓ Added 'c64-knowledge' MCP server")

# Step 6: Write config file
print(f"\n4. Writing configuration...")

try:
    # Create backup if file exists
    if config_file.exists():
        backup_file = config_file.with_suffix('.json.backup')
        import shutil
        shutil.copy2(config_file, backup_file)
        print(f"   ✓ Created backup: {backup_file}")

    # Write new config
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(existing_config, f, indent=2)

    print(f"   ✓ Configuration written to: {config_file}")

except Exception as e:
    print(f"   ✗ Error writing config: {e}")
    sys.exit(1)

# Step 7: Display the configuration
print(f"\n5. Configuration Summary")
print("=" * 70)
print(json.dumps(existing_config, indent=2))

# Step 8: Next steps
print("\n" + "=" * 70)
print("✓ Setup Complete!")
print("=" * 70)
print("\nNext Steps:")
print("1. Restart Claude Desktop (close it completely and reopen)")
print("2. Look for the MCP server indicator (should show 'c64-knowledge')")
print("3. Try asking Claude:")
print("   - 'Search the C64 knowledge base for VIC-II sprites'")
print("   - 'What are the SID chip registers?'")
print("   - 'Show me information about 6502 assembly'")
print("\nIf you see errors:")
print("- Check Claude Desktop's logs")
print("- Verify the paths in the config are correct")
print("- Make sure Claude Desktop has permissions to run Python")

print(f"\nConfig file location: {config_file}")
