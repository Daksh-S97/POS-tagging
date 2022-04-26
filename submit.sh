#!/bin/bash
test -z "$1" && echo "No parameter received. Please call this file with assignment index. Usage example: $0 1" && exit 1

#check no uncommitted changes.
(git status | grep -q modified:) &&  echo  'Error. There are uncommitted changes in your working directory. You can do "git status" to see them.
Please commit or stash uncommitted changes before submitting' && exit 1
COMMIT=$(git log | head -n 1 |  cut -b 1-14)
if (git tag a$1-final 2>/dev/null)
then
    echo "Created tag 'a$1-final' pointing to $COMMIT"
else
    git tag -d a$1-final && git tag a$1-final
    echo "Re-creating tag 'a$1-final'... (now $COMMIT)"
fi
echo "Now syncing with origin..."
git push origin :refs/tags/a$1-final
git push origin --tags #--atomic
echo "Please verify in gitlab that your tag 'a$1-final' matches what you expect. "
