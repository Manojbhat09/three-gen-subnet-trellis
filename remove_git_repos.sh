#!/bin/bash

# Find all .git directories except the main one
find . -name ".git" -type d | while read git_dir; do
    # Skip the main .git directory
    if [ "$git_dir" != "./.git" ]; then
        echo "Removing Git repository at: $git_dir"
        rm -rf "$git_dir"
    fi
done

# Remove any .gitmodules file if it exists
if [ -f .gitmodules ]; then
    rm .gitmodules
fi

# Initialize a fresh Git repository
git init
git add .
git status 