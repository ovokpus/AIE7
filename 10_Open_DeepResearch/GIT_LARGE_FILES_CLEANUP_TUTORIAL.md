# 🚀 Git Large Files Cleanup Tutorial

## The Problem: GitHub's 100MB File Size Limit 😱

Ever tried to push your awesome data science project to GitHub only to get smacked with this error?

```
remote: error: File your_massive_dataset.csv is 143.84 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
```

Yeah, it's a real party pooper! 🎉💥 But don't worry, we've got your back!

## What Usually Happens? 🤔

Common culprits that cause this issue include:

- Large CSV datasets (often >100MB)
- Machine learning model files (.pkl, .h5, .pth)
- Video/audio files accidentally committed
- Database dumps or backup files
- Compressed archives with data

Even after deleting these files from your working directory, they're still lurking in your git history like digital ghosts! 👻

## The Solution: Git History Surgery 🔧

### Step 1: Install git-filter-repo

First, we need the right tools for the job:

```bash
# On macOS with Homebrew
brew install git-filter-repo

# On Ubuntu/Debian
sudo apt-get install git-filter-repo

# On other systems, you might need:
pip install git-filter-repo
```

### Step 2: Navigate to Your Repository Root

Make sure you're in the right place:

```bash
cd /path/to/your/repository/root
git status  # Confirm you're in a git repo
```

### Step 2.5: Find Large Files in Your History (Optional but Helpful)

Before removing files, it's useful to see what's taking up space:

```bash
# Find the largest files in your git history
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print substr($0,6)}' | \
  sort --numeric-sort --key=2 | \
  tail -20

# Or use this shorter version to find files >50MB
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ && $2 > 52428800 {print substr($0,6)}' | \
  sort --numeric-sort --key=2
```

### Step 3: Remove Large Files from Git History

This is where the magic happens! 🪄

```bash
git filter-repo \
  --path "path/to/large/file1.csv" \
  --path "path/to/large/file2.csv" \
  --path "path/to/large/file3.csv" \
  --invert-paths \
  --force
```

**What this command does:**
- `--path`: Specifies which files to target
- `--invert-paths`: Means "remove these paths" (instead of keeping only these paths)
- `--force`: Overrides safety checks (use with caution!)

### Step 4: Re-add Your Remote Origin

`git filter-repo` removes your remote for safety. Add it back:

```bash
git remote add origin git@github.com:yourusername/yourrepo.git
```

### Step 5: Force Push Your Clean History

```bash
git push origin your-branch-name --force
```

## Real-World Example 💪

Here's a typical workflow to clean up large files:

```bash
# Navigate to your repo root
cd /path/to/your/repository

# First, identify large files in your history
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print substr($0,6)}' | sort --numeric-sort --key=2 | tail -10

# Remove the problematic files from history
git filter-repo \
  --path "data/large_dataset.csv" \
  --path "models/huge_model.pkl" \
  --path "videos/presentation.mp4" \
  --invert-paths \
  --force

# Re-add your remote (replace with your actual repo URL)
git remote add origin git@github.com:username/repository.git

# Push the cleaned history
git push origin main --force
```

## Prevention: .gitignore is Your Friend! 🛡️

To avoid this mess in the future, add patterns to your `.gitignore`:

```gitignore
# Large data files
*.csv
*.parquet
*.hdf5
*.h5
*.pkl
*.pickle
*.pth
*.pt
*.safetensors

# Media files
*.mp4
*.avi
*.mov
*.mkv
*.mp3
*.wav

# Archives and compressed files
*.zip
*.tar.gz
*.rar
*.7z

# Common data directories
data/
datasets/
models/
checkpoints/
logs/
cache/

# Database files
*.db
*.sqlite
*.sql

# But keep small example files if needed
!example_small.csv
!sample_data.json
```

## Alternative Solutions 🔄

### Option 1: Git LFS (Large File Storage)
For files you actually need to track:

```bash
git lfs install
git lfs track "*.csv"
git add .gitattributes
```

### Option 2: External Storage
- Store large files in cloud storage (S3, Google Drive, etc.)
- Include download scripts in your repo
- Document where to get the data in your README

### Option 3: Data Version Control (DVC)
```bash
pip install dvc
dvc init
dvc add large_dataset.csv
git add large_dataset.csv.dvc .gitignore
```

## Pro Tips! 💡

1. **Always backup before running git filter-repo** - it rewrites history permanently!
2. **Coordinate with your team** - force pushing affects everyone
3. **Use file size checks** before committing:
   ```bash
   find . -size +50M -type f
   ```
4. **Consider splitting large datasets** into smaller chunks
5. **Document your data sources** so others can reproduce your work

## Troubleshooting 🔍

### "git filter-repo not found"
- Make sure it's installed and in your PATH
- Try the full path: `/usr/local/bin/git-filter-repo`

### "Refusing to destructively overwrite repo history"
- Add the `--force` flag (but be careful!)
- Make sure you're in the repo root

### "Remote rejected"
- You might need `--force` on your push
- Check if branch protection rules are blocking you

## The Happy Ending 🎊

After running these commands, your repository should be squeaky clean and ready to push to GitHub without any complaints about large files!

Remember: Git is powerful but with great power comes great responsibility. Always make backups and communicate with your team before rewriting history!

---

*Created with ❤️ to help fellow developers escape the large file nightmare!*