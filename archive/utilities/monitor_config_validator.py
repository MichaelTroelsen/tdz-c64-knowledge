#!/usr/bin/env python3
"""
Configuration Validator for URL Monitoring

Validates monitor_config.json against expected schema and provides helpful error messages.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_monitor_config(config_path: str = "monitor_config.json") -> Tuple[Dict[str, Any], List[str]]:
    """Validate monitoring configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (validated_config, warnings)

    Raises:
        ConfigValidationError: If configuration is invalid
    """
    warnings = []

    # Check file exists
    if not os.path.exists(config_path):
        raise ConfigValidationError(f"Configuration file not found: {config_path}")

    # Load JSON
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigValidationError(f"Invalid JSON in {config_path}: {e}")

    # Validate data_dir
    if 'data_dir' in config:
        data_dir = os.path.expanduser(config['data_dir'])
        if not os.path.exists(data_dir):
            warnings.append(f"Data directory does not exist: {data_dir}")

    # Validate daily settings
    if 'daily' in config:
        daily = config['daily']
        if not isinstance(daily, dict):
            raise ConfigValidationError("'daily' must be an object")

        # Check enabled flag
        if 'enabled' in daily and not isinstance(daily['enabled'], bool):
            raise ConfigValidationError("'daily.enabled' must be a boolean")

        # Check auto_rescrape
        if 'auto_rescrape' in daily and not isinstance(daily['auto_rescrape'], bool):
            raise ConfigValidationError("'daily.auto_rescrape' must be a boolean")

        # Check notify
        if 'notify' in daily and not isinstance(daily['notify'], bool):
            raise ConfigValidationError("'daily.notify' must be a boolean")

        # Check schedule
        if 'schedule' in daily:
            schedule = daily['schedule']
            if not isinstance(schedule, dict):
                raise ConfigValidationError("'daily.schedule' must be an object")

            if 'time' in schedule:
                time_str = schedule['time']
                if not isinstance(time_str, str):
                    raise ConfigValidationError("'daily.schedule.time' must be a string")
                # Validate time format (HH:MM)
                try:
                    hours, mins = time_str.split(':')
                    h, m = int(hours), int(mins)
                    if not (0 <= h < 24 and 0 <= m < 60):
                        raise ValueError
                except (ValueError, AttributeError):
                    raise ConfigValidationError(
                        f"'daily.schedule.time' must be in HH:MM format, got: {time_str}"
                    )

    # Validate weekly settings (similar to daily)
    if 'weekly' in config:
        weekly = config['weekly']
        if not isinstance(weekly, dict):
            raise ConfigValidationError("'weekly' must be an object")

        # Check enabled
        if 'enabled' in weekly and not isinstance(weekly['enabled'], bool):
            raise ConfigValidationError("'weekly.enabled' must be a boolean")

        # Check check_structure
        if 'check_structure' in weekly and not isinstance(weekly['check_structure'], bool):
            raise ConfigValidationError("'weekly.check_structure' must be a boolean")

        # Check schedule
        if 'schedule' in weekly and 'day' in weekly['schedule']:
            day = weekly['schedule']['day']
            valid_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            if day not in valid_days:
                raise ConfigValidationError(
                    f"'weekly.schedule.day' must be one of {valid_days}, got: {day}"
                )

    # Validate notifications
    if 'notifications' in config:
        notifications = config['notifications']
        if not isinstance(notifications, dict):
            raise ConfigValidationError("'notifications' must be an object")

        # Validate email settings
        if 'email' in notifications:
            email = notifications['email']
            if email.get('enabled'):
                required_fields = ['smtp_server', 'smtp_port', 'from_address', 'to_addresses']
                for field in required_fields:
                    if field not in email:
                        warnings.append(f"Email enabled but '{field}' not configured")

                if 'smtp_port' in email:
                    port = email['smtp_port']
                    if not isinstance(port, int) or not (1 <= port <= 65535):
                        raise ConfigValidationError(
                            f"'notifications.email.smtp_port' must be 1-65535, got: {port}"
                        )

        # Validate Slack settings
        if 'slack' in notifications:
            slack = notifications['slack']
            if slack.get('enabled') and 'webhook_url' not in slack:
                warnings.append("Slack enabled but 'webhook_url' not configured")

    # Validate output settings
    if 'output' in config:
        output = config['output']
        if not isinstance(output, dict):
            raise ConfigValidationError("'output' must be an object")

        if 'keep_days' in output:
            keep_days = output['keep_days']
            if not isinstance(keep_days, int) or keep_days < 0:
                raise ConfigValidationError(
                    f"'output.keep_days' must be a non-negative integer, got: {keep_days}"
                )

    # Validate logging settings
    if 'logging' in config:
        logging_cfg = config['logging']
        if not isinstance(logging_cfg, dict):
            raise ConfigValidationError("'logging' must be an object")

        if 'level' in logging_cfg:
            level = logging_cfg['level']
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if level not in valid_levels:
                raise ConfigValidationError(
                    f"'logging.level' must be one of {valid_levels}, got: {level}"
                )

        if 'max_size_mb' in logging_cfg:
            size = logging_cfg['max_size_mb']
            if not isinstance(size, (int, float)) or size <= 0:
                raise ConfigValidationError(
                    f"'logging.max_size_mb' must be a positive number, got: {size}"
                )

    return config, warnings


def validate_and_load_config(config_path: str = "monitor_config.json", verbose: bool = True) -> Dict[str, Any]:
    """Validate and load configuration with user-friendly output.

    Args:
        config_path: Path to configuration file
        verbose: Print validation results

    Returns:
        Validated configuration dictionary

    Raises:
        ConfigValidationError: If validation fails
    """
    try:
        config, warnings = validate_monitor_config(config_path)

        if verbose:
            print(f"[OK] Configuration loaded from {config_path}")

            if warnings:
                print(f"\n[WARNING] Found {len(warnings)} configuration warnings:")
                for warning in warnings:
                    print(f"  - {warning}")

        return config

    except ConfigValidationError as e:
        if verbose:
            print(f"\n[ERROR] Configuration validation failed:")
            print(f"  {e}")
            print(f"\nPlease fix {config_path} and try again.")
        raise


def main():
    """Standalone config validator."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate monitor_config.json")
    parser.add_argument(
        '--config',
        default='monitor_config.json',
        help='Path to config file (default: monitor_config.json)'
    )
    args = parser.parse_args()

    try:
        config, warnings = validate_monitor_config(args.config)

        print(f"[OK] Configuration is valid!")
        print(f"  File: {args.config}")

        if warnings:
            print(f"\n[WARNING] Warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"  - {warning}")

        # Print summary
        print(f"\nConfiguration summary:")
        if 'data_dir' in config:
            print(f"  Data directory: {config['data_dir']}")
        if 'daily' in config and config['daily'].get('enabled'):
            print(f"  Daily monitoring: Enabled")
        if 'weekly' in config and config['weekly'].get('enabled'):
            print(f"  Weekly monitoring: Enabled")

        return 0

    except ConfigValidationError as e:
        print(f"\n[ERROR] Validation failed:")
        print(f"  {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
