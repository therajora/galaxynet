#!/usr/bin/env python3
"""
Script to manage trained model versions.
"""

import argparse
import sys
from pathlib import Path

# Add root directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from results.version_manager import get_version_manager


def list_versions(args):
    """List model versions."""
    manager = get_version_manager()
    
    if args.model_name:
        versions = manager.list_versions(args.model_name)
        if not versions:
            print(f"No versions found for model: {args.model_name}")
            return
    else:
        versions = manager.list_versions()
        if not versions:
            print("No versions found.")
            return
    
    manager.print_versions_summary(args.model_name)


def show_version(args):
    """Show detailed information for a version."""
    manager = get_version_manager()
    version_info = manager.get_version_info(args.version_id)
    
    if not version_info:
        print(f"Version not found: {args.version_id}")
        return
    
    print("\n" + "=" * 60)
    print(f"VERSION INFORMATION: {args.version_id}")
    print("=" * 60)
    
    for key, value in version_info.items():
        if key == 'metrics' and value:
            print(f"{key}:")
            for metric_key, metric_value in value.items():
                if isinstance(metric_value, float):
                    if 'accuracy' in metric_key.lower():
                        print(f"  {metric_key}: {metric_value:.2%}")
                    else:
                        print(f"  {metric_key}: {metric_value:.4f}")
                else:
                    print(f"  {metric_key}: {metric_value}")
        else:
            print(f"{key}: {value}")


def cleanup_versions(args):
    """Remove old versions."""
    manager = get_version_manager()
    
    if args.model_name:
        manager.cleanup_old_versions(args.model_name, args.keep_count)
    else:
        # List all models and clean each one
        all_versions = manager.list_versions()
        model_names = set(version['model_name'] for version in all_versions)
        
        for model_name in model_names:
            print(f"\nCleaning versions for model: {model_name}")
            manager.cleanup_old_versions(model_name, args.keep_count)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Model version manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List versions')
    list_parser.add_argument('--model-name', type=str, 
                           help='Model name (optional)')
    list_parser.set_defaults(func=list_versions)
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show version information')
    show_parser.add_argument('version_id', type=str, 
                           help='Version ID')
    show_parser.set_defaults(func=show_version)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Remove old versions')
    cleanup_parser.add_argument('--model-name', type=str,
                              help='Model name (optional, cleans all if not specified)')
    cleanup_parser.add_argument('--keep-count', type=int, default=5,
                              help='Number of versions to keep (default: 5)')
    cleanup_parser.set_defaults(func=cleanup_versions)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
