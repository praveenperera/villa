#!/usr/bin/env bash
# Put the Python tools installer inside VC3D.app (Contents/Resources, where VC3D runs it from its install prompt), set
# the app's macOS floor, sign it, and put it in a DMG with an Applications link.
#   sign-app.sh APP INSTALLER OUT TAG
# APP: build-app.sh's VC3D.app; INSTALLER: install-python-tools.sh from make-assets.sh; TAG: the release tag that
# make-assets.sh got, such as vc3d-mac-2026.11.25. Writes OUT/VC3D-mac.app and OUT/VC3D-mac-VERSION.dmg, VERSION being
# TAG without its vc3d-mac- prefix.
#
# With SIGN_IDENTITY set to a Developer ID Application identity in the keychain, it signs with the hardened runtime,
# every nested Mach-O first, then notarizes and staples the app, and signs, notarizes and staples the DMG, so the app
# passes Gatekeeper whether it is opened from the DMG or after it is copied out, online or not. Notarization needs an
# App Store Connect API key: NOTARY_KEY_PATH (the .p8 file), NOTARY_KEY_ID and NOTARY_ISSUER_ID.
# Without SIGN_IDENTITY it signs ad hoc and notarizes nothing. Such a DMG works, but macOS quarantines a downloaded copy,
# so the first launch needs right-click > Open (or xattr -dr com.apple.quarantine VC3D-mac.app).
#
# The DMG holds nothing else to double-click: a quarantined script there would get Gatekeeper's own malware check, which
# a script cannot pass, since only bundles and disk images can be notarized
set -euo pipefail
app_in=$1; installer=$2; out=$3; tag=$4
[[ $tag == vc3d-mac-* ]] || { echo "TAG must start with vc3d-mac-, not $tag" >&2; exit 1; }
version=${tag#vc3d-mac-}
# the installer downloads from its release's URL, so an installer made for another tag would fetch the wrong files
release=$(sed -n 's/^RELEASE=//p' "$installer")
[[ $release == https://github.com/*/*/releases/download/"$tag" ]] || {
  echo "$installer is not for $tag: RELEASE=$release" >&2; exit 1; }
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
identity=${SIGN_IDENTITY:-}
if [ -n "$identity" ]; then
  : "${NOTARY_KEY_PATH:?set NOTARY_KEY_PATH, NOTARY_KEY_ID and NOTARY_ISSUER_ID to notarize}"
  : "${NOTARY_KEY_ID:?set NOTARY_KEY_ID}" "${NOTARY_ISSUER_ID:?set NOTARY_ISSUER_ID}"
  echo "signing with $identity"
else
  echo "no SIGN_IDENTITY: signing ad hoc, without notarization"
fi
mkdir -p "$out"; out=$(cd "$out" && pwd)
app=$out/VC3D-mac.app
dmg=$out/VC3D-mac-$version.dmg

notarize() {
  local log status id
  log=$out/notary-$(basename "$1").json
  xcrun notarytool submit "$1" --key "$NOTARY_KEY_PATH" --key-id "$NOTARY_KEY_ID" --issuer "$NOTARY_ISSUER_ID" \
    --wait --timeout 1h --output-format json > "$log"
  status=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["status"])' "$log")
  id=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["id"])' "$log")
  echo "notarization of $(basename "$1"): $status ($id)"
  if [ "$status" != Accepted ]; then
    xcrun notarytool log "$id" --key "$NOTARY_KEY_PATH" --key-id "$NOTARY_KEY_ID" --issuer "$NOTARY_ISSUER_ID" || true
    exit 1
  fi
}

sign() {
  if [ -n "$identity" ]; then
    codesign --force --timestamp --options runtime --sign "$identity" "$@"
  else
    # no hardened runtime: its library validation would refuse the ad-hoc signed dylibs, which carry no team ID
    codesign --force --sign - "$@"
  fi
}

echo "== copy and sign $app_in at $(date)"
rm -rf "$app" "$dmg"
ditto "$app_in" "$app"
# sealed by the app's signature, so running it from VC3D needs no Gatekeeper check of its own
install -m 755 "$installer" "$app/Contents/Resources/install-python-tools.sh"
"$here/check-macho.sh" "$app"
# Finder and `open` refuse an app whose LSMinimumSystemVersion is above the running macOS (error -10825), while dyld
# does not enforce a dylib's LC_BUILD_VERSION minos at load if only the minor version is newer. So the app's floor is
# the release's, the MIN_MACOS that make-assets.sh wrote into the installer, and the installer and the app refuse the
# same systems. A Mach-O built for a newer major version would fail at load, so that stops the signing; one built for a
# newer minor version of the same major is listed
min_macos=$(sed -n 's/^MIN_MACOS=//p' "$installer")
[[ $min_macos =~ ^[0-9]+\.[0-9]+$ ]] || { echo "no MIN_MACOS=X.Y in $installer" >&2; exit 1; }
newer=$(find "$app/Contents" -type f -print0 | xargs -0 file | awk -F': ' '/Mach-O/ {sub(/ \(for architecture .*/, "", $1); print $1}' \
  | sort -u | while IFS= read -r f; do
      v=$(otool -l "$f" | awk '$1 == "minos" {print $2}' | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)
      [ -n "$v" ] && printf '%s %s\n' "$v" "${f#"$app/"}"
    done | awk -v min="$min_macos" '{
      split($1, a, "."); split(min, b, ".")
      if (a[1] + 0 > b[1] + 0 || (a[1] + 0 == b[1] + 0 && (a[2] + 0 > b[2] + 0 || (a[2] + 0 == b[2] + 0 && a[3] + 0 > 0)))) print
    }')
if [ -n "$newer" ]; then
  echo "built for a newer macOS than $min_macos:"; echo "$newer"
  ! awk -v major="${min_macos%%.*}" '{split($1, a, "."); if (a[1] + 0 > major + 0) bad = 1} END {exit !bad}' <<< "$newer" \
    || { echo "a Mach-O needs a newer major macOS than $min_macos" >&2; exit 1; }
fi
/usr/libexec/PlistBuddy -c "Set :LSMinimumSystemVersion $min_macos" "$app/Contents/Info.plist"
echo "LSMinimumSystemVersion $min_macos"
# inside out: loose Mach-O files (dylibs, Qt plugins, the vc_* tools) deepest first, then each framework bundle, then
# the app, whose main executable VC3D is signed with it
while IFS= read -r f; do
  file -b "$f" | grep -q Mach-O && sign "$f"
done < <(find "$app/Contents" -type f -not -path '*.framework/*' -not -path "$app/Contents/MacOS/VC3D" \
  | awk -F/ '{print NF "\t" $0}' | sort -rn | cut -f2-)
find "$app/Contents/Frameworks" -maxdepth 1 -name '*.framework' -print0 | while IFS= read -r -d '' framework; do
  sign "$framework"
done
sign "$app"
codesign --verify --deep --strict --verbose=2 "$app"

if [ -n "$identity" ]; then
  echo "== notarize the app at $(date)"
  ditto -c -k --keepParent "$app" "$out/VC3D-mac.zip"
  notarize "$out/VC3D-mac.zip"
  xcrun stapler staple "$app"
  rm "$out/VC3D-mac.zip"
fi

echo "== DMG at $(date)"
stage=$out/dmg-stage
rm -rf "$stage"; mkdir -p "$stage"
ditto "$app" "$stage/VC3D-mac.app"
ln -s /Applications "$stage/Applications"
hdiutil create -quiet -volname "VC3D mac $version" -srcfolder "$stage" -format UDZO -ov "$dmg"
rm -rf "$stage"
if [ -n "$identity" ]; then
  codesign --force --timestamp --sign "$identity" "$dmg"
  notarize "$dmg"
  xcrun stapler staple "$dmg"
fi

echo "== checks"
# everything a user could double-click in the DMG
mnt=$out/mnt
hdiutil attach -quiet -nobrowse -readonly -mountpoint "$mnt" "$dmg"
trap 'hdiutil detach -quiet "$mnt" 2>/dev/null || true' EXIT
ls -la "$mnt"
[ "$(ls "$mnt" | sort | tr '\n' ' ')" = "Applications VC3D-mac.app " ] || { echo "unexpected DMG contents" >&2; exit 1; }
codesign --verify --deep --strict "$mnt/VC3D-mac.app"
if [ -n "$identity" ]; then
  spctl -a -vv "$mnt/VC3D-mac.app"
  spctl -a -vv "$app"
  spctl -a -t open --context context:primary-signature -vv "$dmg"
  xcrun stapler validate "$app"
  xcrun stapler validate "$dmg"
fi
hdiutil detach -quiet "$mnt"
trap - EXIT
shasum -a 256 "$dmg"
if [ -n "$identity" ]; then
  echo "signed and notarized $dmg at $(date)"
else
  echo "ad-hoc signed $dmg at $(date): not notarized"
fi
