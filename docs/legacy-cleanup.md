# Legacy Cleanup

This modernization branch removes the original monolithic prototype files from the runnable codebase.

Removed categories:

- Hardcoded credentials, phone numbers, local Windows paths, and personal chat IDs.
- Generated audio and image artifacts.
- Duplicate YAMNet/model folders and temporary Code Runner files.
- Large checked-in YOLO and YAMNet weights.
- Experimental `trash/` scripts.

The old implementation remains available in Git history at commit `951fe80` if a specific behavior needs to be ported forward.
