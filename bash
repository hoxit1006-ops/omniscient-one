# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial production deployment"

# Add remote (your GitHub repository)
git remote add origin https://github.com/yourusername/omniscient-one.git

# Push to main branch
git branch -M main
git push -u origin main
