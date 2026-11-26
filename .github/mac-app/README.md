# VC3D Mac app release

The [VC3D Mac app release workflow](../workflows/vc3d-mac-app.yml) builds the downloadable VC3D for Apple-silicon Macs
and its Python tools on a macOS 15 runner, so both run on macOS 15 and later. A release holds:

- `VC3D-mac-VERSION.dmg`: VC3D-mac.app and an Applications link. The app carries the Python tools installer in
  `Contents/Resources`, so the app's signature covers it, and offers to run it the first time Spiral, Lasagna, sheet
  strips, fibre follow or the neural tracer needs its Python environment (or from Help > Install or Update Python
  Tools).
- `install-python-tools.sh`: the same installer, for Terminal. It needs no compiler, Xcode, Homebrew or conda: it
  downloads a pinned uv, which installs Python 3.14 and three environments (`spiral`, `lasagna`, `vesuvius`) into
  `~/Library/Application Support/Vesuvius Challenge/VC3D/python-tools` from the release's files, each checked against
  the sha256 written into the installer.
- the cp314 macOS arm64 wheels of villa's native packages (vc-reader, vc-object-store, spiral, volume-cartographer,
  vesuvius-lasagna, villa-gpu-manager, and posix-ipc, which has no such wheel on PyPI), one hash-pinned
  `requirements-ENV.txt` per environment, and the villa source tarball the tools run from.
- `SHA256SUMS`.

## Releasing

Push a tag named `vc3d-mac-YYYY.MM.DD`, or `vc3d-mac-YYYY.MM.DD-N` for a later release the same day:

```sh
git tag vc3d-mac-2026.11.26 && git push origin vc3d-mac-2026.11.26
```

The workflow builds the app and the wheels, checks both with Homebrew hidden, signs, and publishes a prerelease under
that tag. The installer downloads from `https://github.com/<this repository>/releases/download/<tag>`, so the tag is
fixed when the files are made; a release under another tag needs a new run. Running the workflow by hand (Actions >
VC3D Mac app release > Run workflow, with a `release_tag`) does everything except publishing and leaves the files in
the run's `VC3D-mac-release` artifact.

A release created by this workflow's token starts no other workflow, so the VC3D release workflows that run on
`release: published` do not attach their packages to it.

## Signing and notarization

Set these repository secrets to sign with a Developer ID and notarize the app and the DMG. Set all five or none.

| Secret | Value |
| --- | --- |
| `MACOS_CERTIFICATE_P12` | base64 of a `.p12` export of the Developer ID Application certificate and its private key (`base64 -i cert.p12`) |
| `MACOS_CERTIFICATE_PASSWORD` | the `.p12`'s password |
| `NOTARY_API_KEY` | the contents of an App Store Connect API key file (`AuthKey_XXXX.p8`) with the Developer role |
| `NOTARY_API_KEY_ID` | that key's ID |
| `NOTARY_API_ISSUER_ID` | the issuer ID shown with the key |

Without them the workflow still builds everything, signs the app ad hoc, and warns that the build is not notarized.
Such a DMG works, but macOS quarantines a downloaded copy, so its first launch needs right-click > Open.

The optional repository variable `VC3D_MAC_RUNNER` names another macOS 15 arm64 runner to use instead of `macos-15`.

## Scripts

- `build-app.sh OUT [build_macos.sh options]`: builds VC3D with `scripts/build_macos.sh`, packages `OUT/VC3D.app`, and
  checks it: one libomp, no leaked paths, nothing built for a newer major macOS than `MIN_MACOS`.
- `check-macho.sh DIR...`: fails if any Mach-O loads a library, or searches an rpath, in Homebrew, `/usr/local`, a home
  directory or a temporary directory.
- `sign-app.sh APP INSTALLER OUT TAG`: puts the installer in the app, sets `LSMinimumSystemVersion` to the installer's
  `MIN_MACOS`, signs, and builds the DMG. With `SIGN_IDENTITY`, `NOTARY_KEY_PATH`, `NOTARY_KEY_ID` and
  `NOTARY_ISSUER_ID` set it signs with the hardened runtime and notarizes and staples the app and the DMG; without
  them it signs ad hoc.
- `build-opencv.sh PREFIX [JOBS]`: a static OpenCV core and imgproc for the vc-reader, vc-object-store and spiral
  wheels, so they bundle no OpenCV dylibs.
- `build-wheels.sh SRC OUT [JOBS]`: builds the wheels, delocates them, and points every libomp reference at torch's
  copy, so a process holds one OpenMP runtime (`fix_wheel.py` does the rewrite and the leak checks).
- `make-assets.sh VILLA WHEELS OUT REPOSITORY TAG MIN_MACOS`: the source tarball, the locks (resolved from
  `lock-inputs/*.in` for `MIN_MACOS`) and the filled-in installer.
- `install-python-tools.sh.in`: the installer template. `VC3D_PYTHON_TOOLS` overrides the install folder and
  `VC3D_TOOLS_ASSETS` names a local folder of release files to use instead of downloading them.
- `omp_check.py MODULE...`: imports the modules and torch, and fails unless exactly one OpenMP runtime (torch's) is
  loaded and it runs in parallel.

To build a release on a Mac by hand:

```sh
R=.github/mac-app
$R/build-app.sh /tmp/vc3d-app --install-deps
$R/build-opencv.sh /tmp/opencv
OCV=/tmp/opencv TORCH_LIBOMPS="<torch/lib/libomp.dylib of each torch the locks pin>" \
  MACOSX_DEPLOYMENT_TARGET=15.0 $R/build-wheels.sh <git archive of HEAD> /tmp/wheels
$R/make-assets.sh . /tmp/wheels/wheels /tmp/release ScrollPrize/villa vc3d-mac-2026.11.26 15.0
$R/sign-app.sh /tmp/vc3d-app/VC3D.app /tmp/release/install-python-tools.sh /tmp/signed vc3d-mac-2026.11.26
```

To check an install as a new user gets it, with only the system's tools:

```sh
home=$(mktemp -d)
env -i HOME="$home" PATH=/usr/bin:/bin:/usr/sbin:/sbin VC3D_TOOLS_ASSETS=/tmp/release \
  bash /tmp/release/install-python-tools.sh
```
