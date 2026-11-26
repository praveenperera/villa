#!/usr/bin/env bash
# Fail if any Mach-O under DIR loads a library, or searches an rpath, in Homebrew, /usr/local, a home directory or a
# temporary directory. Prints each leak, then how many Mach-O files it checked
#   check-macho.sh DIR...
set -euo pipefail
leak='/opt/homebrew|/usr/local|/Users/|/private/|/tmp/|/opt/llvm'
n=0; bad=0
while IFS= read -r -d '' f; do
  file -b "$f" | grep -q 'Mach-O' || continue
  n=$((n + 1))
  deps=$(otool -L "$f" | tail -n +2 | sed 's/ (compatibility.*//; s/^[[:space:]]*//')
  rp=$(otool -l "$f" | awk '/cmd LC_RPATH/ {getline; getline; print $2}')
  hits=$(printf '%s\n%s\n' "$deps" "$rp" | grep -E "$leak" || true)
  if [ -n "$hits" ]; then
    bad=$((bad + 1))
    printf 'LEAK %s:\n%s\n' "${f#"$PWD"/}" "$hits"
  fi
done < <(find "$@" -type f -print0)
echo "checked $n Mach-O files, $bad with leaked paths"
[ "$bad" -eq 0 ]
