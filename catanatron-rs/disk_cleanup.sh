#!/usr/bin/env bash
# Disk cleanup reviewed in chat. Shows every target and its size, then asks
# once before deleting anything. (npm cache and ~/.gradle/caches were already
# cleared.)
#   bash disk_cleanup.sh
set -uo pipefail

ARCHIVE="$HOME/Desktop/prog/archive"

# Single files/folders (irreversible: backups and data).
TARGETS=(
  "$HOME/Downloads/restore-6924a614f703556e3ccf1995.tar.gz"
  "$HOME/Downloads/mongo-final-backup.tar.tar"
  "$HOME/Downloads/restore-68fb83ab4944130b2d9d298b.tar.gz"
  "$ARCHIVE/pbc/final/pbc-athena-data-lake"          # extracted copy; the .tar.gz next to it is kept
  "$ARCHIVE/rob/periscope/epstein"
  "$HOME/Downloads/anki-24.11-linux-qt6"             # older Anki; anki-25.02.5 is kept
  "$HOME/Downloads/anki-25.02 (2).5-linux-qt6"       # duplicate download
)

# Regenerable: node_modules (npm install) and .terraform (terraform init) under archive/.
mapfile -d '' REGEN < <(find "$ARCHIVE" -xdev -type d \( -name .terraform -o -name node_modules \) -prune -print0)

echo "== Files and folders"
for t in "${TARGETS[@]}"; do
  if [[ -e "$t" ]]; then du -sh "$t"; else echo "   (missing) $t"; fi
done
echo
echo "== ${#REGEN[@]} node_modules/.terraform folders under $ARCHIVE"
if (( ${#REGEN[@]} )); then du -csh "${REGEN[@]}" | tail -1; fi
echo
echo "== Unused Docker volumes (not attached to any container, running or stopped)"
docker volume ls -q --filter dangling=true | sed 's/^/   /'
docker system df | grep -i "local volumes"
echo
df -h / | tail -1
read -r -p "Delete everything listed above? [y/N] " answer
[[ "$answer" == [yY] ]] || { echo "Nothing deleted."; exit 0; }

for t in "${TARGETS[@]}"; do
  [[ -e "$t" ]] && rm -rf -- "$t" && echo "removed $t"
done
if (( ${#REGEN[@]} )); then rm -rf -- "${REGEN[@]}" && echo "removed ${#REGEN[@]} node_modules/.terraform folders"; fi
docker volume prune -a -f | tail -1
echo
df -h / | tail -1
