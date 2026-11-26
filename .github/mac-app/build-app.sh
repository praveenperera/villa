#!/usr/bin/env bash
# Build VC3D and the vc_* tools with scripts/build_macos.sh and package them as a self-contained VC3D.app: the
# vc_runtime install component runs macdeployqt, which copies every dylib and Qt plugin in. Then check the bundle:
# exactly one libomp, no Mach-O that loads or searches anything in Homebrew, /usr/local or a build directory
# (check-macho.sh), and none built for a newer major macOS than MIN_MACOS, which dyld would refuse to load there.
# Signing for distribution is sign-app.sh's job
#   build-app.sh OUT [build_macos.sh options...]   such as --install-deps --sccache --jobs 3
# Writes OUT/VC3D.app. MIN_MACOS (default 15.0) is the oldest macOS the release supports
set -euo pipefail
out=$1; shift
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
vc=$(cd "$here/../../volume-cartographer" && pwd)
min_macos=${MIN_MACOS:-15.0}
mkdir -p "$out"; out=$(cd "$out" && pwd)
app=$out/VC3D.app
echo "build from $(git -C "$vc" rev-parse HEAD) at $(date)"

(cd "$vc" && ./scripts/build_macos.sh "$@")

# the install step (scripts/deploy_macos.py) uses Homebrew LLVM's binutils, as the build does
brew_prefix=$(brew --prefix)
export PATH="$brew_prefix/opt/llvm/bin:$PATH"
rm -rf "$app"
cmake --install "$vc/build-macos" --prefix "$out" --component vc_runtime
[ -d "$app" ] || { echo "no VC3D.app in $out" >&2; exit 1; }
for bin in VC3D flatboi vc_gen_normalgrids vc_ngrids vc_render_tifxyz vc_grow_seg_from_seed vc_tifxyz2obj; do
  [ -x "$app/Contents/MacOS/$bin" ] || { echo "missing Contents/MacOS/$bin" >&2; exit 1; }
done

# a second OpenMP runtime in the process aborts it at the first parallel region (OMP Error #15)
libomps=$(find "$app" -name 'libomp*')
echo "libomp in the bundle: $libomps"
[ "$(printf '%s\n' "$libomps" | grep -c .)" -eq 1 ] || { echo "expected exactly one libomp" >&2; exit 1; }

"$here/check-macho.sh" "$app"

# dyld refuses a Mach-O built for a newer major macOS; a newer minor version of the same major still loads
minos=$(find "$app/Contents" -type f -print0 | xargs -0 file | awk -F': ' '/Mach-O/ {sub(/ \(for architecture .*/, "", $1); print $1}' \
  | sort -u | while IFS= read -r f; do otool -l "$f" | awk '$1 == "minos" {print $2}'; done \
  | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)
echo "newest minos in the bundle: $minos"
[ "${minos%%.*}" -le "${min_macos%%.*}" ] || { echo "a Mach-O needs macOS $minos, above $min_macos" >&2; exit 1; }

codesign --verify --deep --strict "$app"
echo "built $app at $(date)"
