# Data Access

EmoVerse data files are intentionally not stored in this Git repository.

The public repository should contain only lightweight release materials such as documentation, loaders, metadata examples, citation files, and project pages. Raw images, generated images, annotations, masks, archives, spreadsheets, paper drafts, and experiment outputs should stay outside Git.

## Planned Release Flow

1. Publish the repository homepage and documentation.
2. Finalize the dataset license and data use terms.
3. Upload release packages to an external storage service.
4. Add checksums, split files, and loader examples.
5. Announce the download URL in this file and on the project page.

## Local Safeguards

Before pushing to GitHub, run:

```powershell
pwsh ./scripts/check_release.ps1
```

The script reports large files and common dataset artifacts that should not be tracked.

## Files That Should Not Be Committed

- Raw image folders
- Annotation folders
- Mask folders
- `.zip`, `.tar`, `.tar.gz`, `.7z`, `.rar`
- `.xlsx`, `.xls`, `.csv` unless explicitly sanitized for release
- `.pdf`, `.pptx`, `.psd`, `.ai`
- Model checkpoints and experiment outputs

## Data License

Dataset license terms are pending final review. Until the public release terms are posted, do not redistribute local dataset files.
