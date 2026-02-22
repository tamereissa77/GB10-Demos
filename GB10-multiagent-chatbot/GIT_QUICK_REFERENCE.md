# Git Quick Reference Card

## 🔗 Your Repository

**URL**: https://github.com/tamereissa77/GB10-multiagent-chatbot.git
**Branch**: main
**Local Path**: `d:\vscodes\GB10\multi-agent-chatbot`

---

## 📝 Daily Git Workflow

### 1. Check Status
```bash
git status
```

### 2. Add Changes
```bash
# Add all changes
git add .

# Or add specific files
git add filename.txt
```

### 3. Commit Changes
```bash
git commit -m "Brief description of changes"
```

### 4. Push to GitHub
```bash
git push
```

### 5. Pull Latest Changes
```bash
git pull
```

---

## 🚀 Common Commands

### View Changes
```bash
# See what changed
git diff

# See commit history
git log --oneline

# See last 5 commits
git log --oneline -n 5
```

### Undo Changes
```bash
# Discard changes in a file
git checkout -- filename.txt

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

### Branches
```bash
# Create new branch
git checkout -b feature/new-feature

# Switch branches
git checkout main

# List all branches
git branch -a

# Delete branch
git branch -d feature/old-feature
```

### Remote Operations
```bash
# Check remote
git remote -v

# Fetch updates
git fetch origin

# Pull and merge
git pull origin main

# Push branch
git push origin branch-name
```

---

## 💡 Quick Tips

### Before Making Changes
```bash
git pull  # Always pull first
```

### After Making Changes
```bash
git add .
git commit -m "Description"
git push
```

### If Push Fails
```bash
git pull --rebase
git push
```

---

## 🔧 Useful Aliases (Optional)

Add to your `.gitconfig`:

```bash
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
```

Then use:
```bash
git st      # instead of git status
git co main # instead of git checkout main
git br      # instead of git branch
git ci -m "message" # instead of git commit -m "message"
```

---

## 📊 Repository Info

```bash
# View remote URL
git remote -v

# View current branch
git branch

# View all commits
git log

# View file changes
git diff filename.txt
```

---

## 🆘 Emergency Commands

### Accidentally Committed Wrong Files
```bash
git reset --soft HEAD~1
# Fix the files
git add correct-files
git commit -m "Correct commit"
```

### Need to Discard All Local Changes
```bash
git reset --hard HEAD
git clean -fd
```

### Merge Conflict
```bash
# After pulling
# 1. Edit conflicted files
# 2. Remove conflict markers
# 3. Then:
git add .
git commit -m "Resolve conflicts"
git push
```

---

## 📱 GitHub Web Interface

Visit: https://github.com/tamereissa77/GB10-multiagent-chatbot

You can:
- View all files
- Edit files directly
- Create issues
- Review commits
- Manage branches
- Add collaborators

---

## ✅ Best Practices

1. **Commit Often**: Small, frequent commits are better
2. **Write Clear Messages**: Describe what and why
3. **Pull Before Push**: Always pull latest changes first
4. **Test Before Commit**: Ensure code works
5. **Use Branches**: For new features or experiments

---

## 📝 Commit Message Examples

Good commit messages:
```bash
git commit -m "Add Dell logo to header"
git commit -m "Fix: Resolve memory leak in agent.py"
git commit -m "Update: Improve RAG query performance"
git commit -m "Docs: Add installation troubleshooting"
```

Bad commit messages:
```bash
git commit -m "fix"
git commit -m "updates"
git commit -m "stuff"
```

---

## 🔄 Typical Workflow

```bash
# Start of day
cd d:\vscodes\GB10\multi-agent-chatbot
git pull

# Make changes to files...

# Check what changed
git status
git diff

# Stage and commit
git add .
git commit -m "Descriptive message"

# Push to GitHub
git push

# End of day - verify
git status  # Should show "nothing to commit, working tree clean"
```

---

**Quick Help**: Run `git --help` or `git <command> --help` for more info
