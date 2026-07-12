#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_ROOT="$ROOT/apps/MTPLXApp"
BUILD_SCRIPT="$APP_ROOT/script/build_and_run.sh"
VERSION="${MTPLX_RELEASE_VERSION:-$(/usr/bin/awk -F'"' '/^version = / { print $2; exit }' "$ROOT/pyproject.toml")}"
APP_BUILD="${MTPLX_RELEASE_BUILD:-10000}"
RELEASE_TAG="${MTPLX_RELEASE_TAG:-v$VERSION}"
GITHUB_REPO="${MTPLX_GITHUB_REPO:-youssofal/mtplx}"
GITHUB_ASSET_BASE="${MTPLX_GITHUB_ASSET_BASE:-https://github.com/$GITHUB_REPO/releases/download/$RELEASE_TAG}"
OUT_ROOT="${MTPLX_RELEASE_OUT:-$HOME/.mtplx/releases/$VERSION-$(/bin/date -u +%Y%m%dT%H%M%SZ)}"
APP_BUNDLE="$OUT_ROOT/MTPLX.app"
DMG="$OUT_ROOT/MTPLX-$VERSION.dmg"
DMG_STAGE="$OUT_ROOT/dmg-stage"
PYTHON_DIST="$OUT_ROOT/python"
PYTOOLS_VENV="$OUT_ROOT/python-tools-venv"
SITE_OUT="$OUT_ROOT/site"
RELEASES_OUT="$SITE_OUT/releases"
NOTES_OUT="$RELEASES_OUT/notes"
SPARKLE_ARCHIVES="$OUT_ROOT/sparkle-archives"
APP_NOTARY_ZIP="$OUT_ROOT/MTPLX-$VERSION.app.zip"

# Guard against accidentally shipping a pre-1.0 or malformed version.
if [[ ! "$VERSION" =~ ^[1-9][0-9]*\.[0-9]+\.[0-9]+$ ]]; then
  echo "error: release version must be a stable x.y.z (>= 1.0.0), got $VERSION" >&2
  exit 1
fi

export MTPLX_SPARKLE_PUBLIC_ED_KEY="${MTPLX_SPARKLE_PUBLIC_ED_KEY:-GQ0sTm6nb5kv+Btri7wc4LqnXGZ48vIs6PGMwsI/mBM=}"
SPARKLE_PRIVATE_KEY="${MTPLX_SPARKLE_PRIVATE_KEY:-${SPARKLE_PRIVATE_KEY:-}}"
SPARKLE_PRIVATE_KEY_FILE="${MTPLX_SPARKLE_PRIVATE_KEY_FILE:-${SPARKLE_PRIVATE_KEY_FILE:-}}"
SPARKLE_KEY_ACCOUNT="${MTPLX_SPARKLE_KEY_ACCOUNT:-ed25519}"

CODESIGN_IDENTITY="${MTPLX_CODESIGN_IDENTITY:-${MTPLX_DEVELOPER_ID_APPLICATION:-}}"
if [[ -z "$CODESIGN_IDENTITY" ]]; then
  echo "error: set MTPLX_CODESIGN_IDENTITY or MTPLX_DEVELOPER_ID_APPLICATION to the Developer ID Application identity" >&2
  exit 1
fi

if [[ -z "$SPARKLE_PRIVATE_KEY" && -z "$SPARKLE_PRIVATE_KEY_FILE" && "${MTPLX_SPARKLE_ALLOW_KEYCHAIN:-0}" != "1" ]]; then
  echo "error: Sparkle appcast signing key missing; set MTPLX_SPARKLE_PRIVATE_KEY or MTPLX_SPARKLE_PRIVATE_KEY_FILE" >&2
  echo "note: set MTPLX_SPARKLE_ALLOW_KEYCHAIN=1 only for interactive local signing, because Keychain access can block unattended releases" >&2
  exit 1
fi

if [[ "${MTPLX_RELEASE_ALLOW_DIRTY:-0}" != "1" ]]; then
  if ! (cd "$ROOT" && git diff --quiet && git diff --cached --quiet); then
    echo "error: release checkout has uncommitted tracked changes" >&2
    exit 1
  fi
  if [[ -n "$(cd "$ROOT" && git ls-files --others --exclude-standard)" ]]; then
    echo "error: release checkout has untracked files" >&2
    exit 1
  fi
fi

mkdir -p "$OUT_ROOT" "$PYTHON_DIST" "$SITE_OUT" "$RELEASES_OUT" "$NOTES_OUT" "$SPARKLE_ARCHIVES"

# Release gates: the full Python and Swift suites must be green before any
# artifact is built or signed. MTPLX_RELEASE_SKIP_TESTS=1 exists only for
# rehearsals of the packaging pipeline itself.
if [[ "${MTPLX_RELEASE_SKIP_TESTS:-0}" != "1" ]]; then
  if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "error: $ROOT/.venv is missing; create it and install '.[server,dev]' before releasing" >&2
    exit 1
  fi
  echo "Release gate: pytest"
  (cd "$ROOT" && "$ROOT/.venv/bin/python" -m pytest -q)
  echo "Release gate: swift test"
  (cd "$APP_ROOT" && swift test)
else
  echo "warning: MTPLX_RELEASE_SKIP_TESTS=1 — test gates skipped; this artifact is not release-ready" >&2
fi

RELEASE_NOTES_MD="$ROOT/docs/releases/v$VERSION.md"
if [[ ! -f "$RELEASE_NOTES_MD" ]]; then
  echo "error: release notes source missing: $RELEASE_NOTES_MD" >&2
  echo "write the user-facing notes for v$VERSION before releasing" >&2
  exit 1
fi

submit_notarization() {
  local artifact="$1"
  if [[ -n "${MTPLX_NOTARY_PROFILE:-}" ]]; then
    xcrun notarytool submit "$artifact" --keychain-profile "$MTPLX_NOTARY_PROFILE" --wait
  elif [[ -n "${MTPLX_ASC_KEY:-}" && -n "${MTPLX_ASC_KEY_ID:-}" && -n "${MTPLX_ASC_ISSUER_ID:-}" ]]; then
    xcrun notarytool submit "$artifact" \
      --key "$MTPLX_ASC_KEY" \
      --key-id "$MTPLX_ASC_KEY_ID" \
      --issuer "$MTPLX_ASC_ISSUER_ID" \
      --wait
  else
    echo "error: notarization credentials missing; set MTPLX_NOTARY_PROFILE or App Store Connect key env" >&2
    exit 1
  fi
}

echo "Building Python artifacts for mtplx==$VERSION"
# A stale mtplx.egg-info/SOURCES.txt resurrects files MANIFEST.in excludes
# (it once leaked the internal LOG.md into the sdist); always build from a
# clean manifest.
rm -rf "$ROOT/mtplx.egg-info"
python3 -m venv "$PYTOOLS_VENV"
"$PYTOOLS_VENV/bin/python" -m pip install --upgrade pip build twine markdown
"$PYTOOLS_VENV/bin/python" -m build "$ROOT" --outdir "$PYTHON_DIST"
"$PYTOOLS_VENV/bin/python" -m twine check "$PYTHON_DIST"/*
PYTHON_WHEEL="$PYTHON_DIST/mtplx-$VERSION-py3-none-any.whl"
if [[ ! -f "$PYTHON_WHEEL" ]]; then
  echo "error: expected runtime wheel missing: $PYTHON_WHEEL" >&2
  exit 1
fi

# Bundled Python interpreter: python-build-standalone (Astral), pinned by
# release tag + sha256 so every build ships the exact same interpreter.
# This is what removes the "Install Homebrew" wall on pristine Macs.
PBS_RELEASE="${MTPLX_PBS_RELEASE:-20260602}"
PBS_PYTHON_VERSION="${MTPLX_PBS_PYTHON_VERSION:-3.14.5}"
PBS_ARTIFACT="cpython-$PBS_PYTHON_VERSION+$PBS_RELEASE-aarch64-apple-darwin-install_only_stripped.tar.gz"
PBS_SHA256="${MTPLX_PBS_SHA256:-3a0373cc39fefd494754ef555267f245c720cddbaaabf63a7c9a4269f1e56532}"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_RELEASE/$PBS_ARTIFACT"
PBS_CACHE_DIR="${MTPLX_PBS_CACHE_DIR:-$HOME/.mtplx/build-cache/python-build-standalone}"
PBS_TARBALL="$PBS_CACHE_DIR/$PBS_ARTIFACT"
PBS_EXTRACT_DIR="$OUT_ROOT/python-runtime"

echo "Preparing bundled Python runtime ($PBS_ARTIFACT)"
mkdir -p "$PBS_CACHE_DIR"
if [[ ! -f "$PBS_TARBALL" ]]; then
  /usr/bin/curl -fL --retry 3 -o "$PBS_TARBALL.partial" "$PBS_URL"
  mv "$PBS_TARBALL.partial" "$PBS_TARBALL"
fi
if ! echo "$PBS_SHA256  $PBS_TARBALL" | /usr/bin/shasum -a 256 -c - >/dev/null 2>&1; then
  echo "error: bundled Python tarball failed its sha256 pin: $PBS_TARBALL" >&2
  echo "expected $PBS_SHA256" >&2
  exit 1
fi
rm -rf "$PBS_EXTRACT_DIR"
mkdir -p "$PBS_EXTRACT_DIR"
/usr/bin/tar -xzf "$PBS_TARBALL" -C "$PBS_EXTRACT_DIR" --strip-components 1
if [[ ! -x "$PBS_EXTRACT_DIR/bin/python3" ]]; then
  echo "error: bundled Python extraction failed; $PBS_EXTRACT_DIR/bin/python3 missing" >&2
  exit 1
fi

echo "Building signed MTPLX.app"
MTPLX_APP_PUBLIC_RELEASE=1 \
MTPLX_APP_VERSION="$VERSION" \
MTPLX_APP_BUILD="$APP_BUILD" \
MTPLX_APP_BUNDLE_DIR="$APP_BUNDLE" \
MTPLX_APP_EMBED_LOCAL_RUNTIME_WRAPPER=0 \
MTPLX_RUNTIME_WHEEL="$PYTHON_WHEEL" \
MTPLX_REQUIRE_RUNTIME_WHEEL_RESOURCE=1 \
MTPLX_BUNDLED_PYTHON_DIR="$PBS_EXTRACT_DIR" \
MTPLX_REQUIRE_BUNDLED_PYTHON_RESOURCE=1 \
MTPLX_REQUIRE_THERMALFORGE_RESOURCE=1 \
MTPLX_CODESIGN_IDENTITY="$CODESIGN_IDENTITY" \
"$BUILD_SCRIPT" --no-launch

/usr/bin/codesign --verify --deep --strict --verbose=4 "$APP_BUNDLE"

# Library-validation gate: without this entitlement on the bundled
# interpreter, every pip-installed wheel's linker-signed extension is
# rejected with "different Team IDs" on macOS 15 and earlier and the
# engine dies on its first numpy import on customer Macs (macOS 26
# relaxed the rule, so dev machines never reproduce it).
PYTHON_BIN_DIR="$APP_BUNDLE/Contents/Resources/PythonRuntime/bin"
ENTITLED=0
for candidate in "$PYTHON_BIN_DIR"/python3*; do
  [[ -f "$candidate" && ! -L "$candidate" ]] || continue
  /usr/bin/file "$candidate" | /usr/bin/grep -q 'Mach-O' || continue
  if /usr/bin/codesign -d --entitlements - "$candidate" 2>/dev/null \
    | /usr/bin/grep -q 'disable-library-validation'; then
    ENTITLED=1
  else
    echo "error: $candidate lacks com.apple.security.cs.disable-library-validation; wheels will not load on macOS 15" >&2
    exit 1
  fi
done
if [[ "$ENTITLED" != "1" ]]; then
  echo "error: no bundled python interpreter found to verify the library-validation entitlement" >&2
  exit 1
fi

if /usr/bin/grep -R -I -E '/Users/youssof|[0-9a-f]+-dirty|MTPLXLocalRuntimeWrapperPath' "$APP_BUNDLE" >/dev/null 2>&1; then
  echo "error: signed app contains a local path, dirty marker, or local runtime wrapper key" >&2
  exit 1
fi

if [[ "${MTPLX_SKIP_NOTARIZATION:-0}" != "1" ]]; then
  echo "Submitting app for notarization"
  /usr/bin/ditto -c -k --keepParent --norsrc "$APP_BUNDLE" "$APP_NOTARY_ZIP"
  submit_notarization "$APP_NOTARY_ZIP"
  xcrun stapler staple "$APP_BUNDLE"
  xcrun stapler validate "$APP_BUNDLE"
  /usr/sbin/spctl --assess --type execute --verbose "$APP_BUNDLE"
fi

echo "Creating DMG"
/bin/rm -rf "$DMG_STAGE"
/bin/mkdir -p "$DMG_STAGE"
/usr/bin/ditto --norsrc "$APP_BUNDLE" "$DMG_STAGE/MTPLX.app"
/bin/ln -s /Applications "$DMG_STAGE/Applications"
/usr/bin/xattr -rc "$DMG_STAGE" >/dev/null 2>&1 || true
/usr/bin/find "$DMG_STAGE" -depth -exec /usr/bin/xattr -c {} + >/dev/null 2>&1 || true

/usr/bin/hdiutil create \
  -volname "MTPLX $VERSION" \
  -srcfolder "$DMG_STAGE" \
  -ov \
  -format UDZO \
  "$DMG"

/usr/bin/codesign --force --timestamp --sign "$CODESIGN_IDENTITY" "$DMG"
/usr/bin/codesign --verify --deep --strict --verbose=4 "$APP_BUNDLE"

if [[ "${MTPLX_SKIP_NOTARIZATION:-0}" != "1" ]]; then
  echo "Submitting DMG for notarization"
  submit_notarization "$DMG"
  xcrun stapler staple "$DMG"
  xcrun stapler validate "$DMG"
  /usr/sbin/spctl --assess --type execute --verbose "$APP_BUNDLE"
  /usr/sbin/spctl --assess --type open --context context:primary-signature --verbose "$DMG"
else
  echo "warning: notarization skipped; this artifact is not release-ready" >&2
fi

DMG_SHA256="$(/usr/bin/shasum -a 256 "$DMG" | /usr/bin/awk '{print $1}')"
DMG_SIZE="$(/usr/bin/stat -f%z "$DMG")"
printf '%s  %s\n' "$DMG_SHA256" "$(basename "$DMG")" > "$DMG.sha256"

RELEASE_NOTES_URL="https://mtplx.com/releases/notes/v$VERSION.html"
DMG_URL="$GITHUB_ASSET_BASE/$(basename "$DMG")"

# Sparkle's "what's new" dialog and the hosted notes page render the real
# release notes authored in docs/releases/v$VERSION.md (gated above).
"$PYTOOLS_VENV/bin/python" - "$RELEASE_NOTES_MD" "$NOTES_OUT/v$VERSION.html" "$VERSION" <<'PY'
import pathlib
import sys

import markdown

source, destination, version = sys.argv[1], sys.argv[2], sys.argv[3]
body = markdown.markdown(
    pathlib.Path(source).read_text(encoding="utf-8"),
    extensions=["extra"],
)
html = (
    "<!doctype html>\n"
    '<meta charset="utf-8">\n'
    f"<title>MTPLX {version}</title>\n"
    '<style>body{font-family:-apple-system,system-ui,sans-serif;'
    "max-width:42em;margin:2em auto;padding:0 1em;line-height:1.55}"
    "h1,h2{line-height:1.2}code{background:#f2f2f4;padding:0 .25em;"
    "border-radius:4px}</style>\n"
    f"{body}\n"
)
pathlib.Path(destination).write_text(html, encoding="utf-8")
PY

python3 - "$RELEASES_OUT/latest.json" "$VERSION" "$APP_BUILD" "$DMG_URL" "$DMG_SHA256" "$DMG_SIZE" "$RELEASE_NOTES_URL" <<'PY'
import datetime
import json
import sys

path, version, build, dmg_url, sha, size, notes = sys.argv[1:]
payload = {
    "app_version": version,
    "app_build": build,
    "minimum_cli_version": version,
    "recommended_cli_version": version,
    "dmg_url": dmg_url,
    "dmg_sha256": sha,
    "dmg_size_bytes": int(size),
    "pypi_version": version,
    "homebrew_formula_version": version,
    "release_notes_url": notes,
    "published_at": datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

/usr/bin/ditto --norsrc "$DMG" "$SPARKLE_ARCHIVES/$(basename "$DMG")"
/usr/bin/ditto --norsrc "$NOTES_OUT/v$VERSION.html" "$SPARKLE_ARCHIVES/MTPLX-$VERSION.html"
/usr/bin/ditto --norsrc "$NOTES_OUT/v$VERSION.html" "$NOTES_OUT/MTPLX-$VERSION.html"

GENERATE_APPCAST="$(
  /usr/bin/find "$APP_ROOT/.build" -path '*/bin/generate_appcast' -type f -print 2>/dev/null | /usr/bin/head -n 1
)"
if [[ -z "$GENERATE_APPCAST" ]]; then
  echo "error: Sparkle generate_appcast tool not found under $APP_ROOT/.build" >&2
  exit 1
fi

GENERATE_KEYS="$(
  /usr/bin/find "$APP_ROOT/.build" -path '*/bin/generate_keys' -type f -print 2>/dev/null | /usr/bin/head -n 1
)"
if [[ -z "$GENERATE_KEYS" ]]; then
  echo "error: Sparkle generate_keys tool not found under $APP_ROOT/.build" >&2
  exit 1
fi

if [[ -z "$SPARKLE_PRIVATE_KEY" && -z "$SPARKLE_PRIVATE_KEY_FILE" && "${MTPLX_SPARKLE_ALLOW_KEYCHAIN:-0}" == "1" ]]; then
  KEYCHAIN_PUBLIC_KEY="$("$GENERATE_KEYS" -p --account "$SPARKLE_KEY_ACCOUNT")"
  if [[ "$KEYCHAIN_PUBLIC_KEY" != "$MTPLX_SPARKLE_PUBLIC_ED_KEY" ]]; then
    echo "error: Sparkle Keychain public key does not match MTPLX_SPARKLE_PUBLIC_ED_KEY" >&2
    echo "  keychain[$SPARKLE_KEY_ACCOUNT]: $KEYCHAIN_PUBLIC_KEY" >&2
    echo "  app build public key: $MTPLX_SPARKLE_PUBLIC_ED_KEY" >&2
    exit 1
  fi
fi

APPCAST_ARGS=()
if "$GENERATE_APPCAST" --help 2>&1 | /usr/bin/grep -q -- '--download-url-prefix'; then
  APPCAST_ARGS+=(--download-url-prefix "$GITHUB_ASSET_BASE/")
fi
if "$GENERATE_APPCAST" --help 2>&1 | /usr/bin/grep -q -- '--full-release-notes-url'; then
  APPCAST_ARGS+=(--full-release-notes-url "$RELEASE_NOTES_URL")
fi

if [[ -n "$SPARKLE_PRIVATE_KEY" ]]; then
  printf '%s' "$SPARKLE_PRIVATE_KEY" | "$GENERATE_APPCAST" --ed-key-file - "${APPCAST_ARGS[@]}" "$SPARKLE_ARCHIVES"
elif [[ -n "$SPARKLE_PRIVATE_KEY_FILE" ]]; then
  "$GENERATE_APPCAST" --ed-key-file "$SPARKLE_PRIVATE_KEY_FILE" "${APPCAST_ARGS[@]}" "$SPARKLE_ARCHIVES"
elif [[ "${MTPLX_SPARKLE_ALLOW_KEYCHAIN:-0}" == "1" ]]; then
  "$GENERATE_APPCAST" --account "$SPARKLE_KEY_ACCOUNT" "${APPCAST_ARGS[@]}" "$SPARKLE_ARCHIVES"
else
  echo "error: Sparkle appcast signing key missing; set MTPLX_SPARKLE_PRIVATE_KEY or MTPLX_SPARKLE_PRIVATE_KEY_FILE" >&2
  echo "note: set MTPLX_SPARKLE_ALLOW_KEYCHAIN=1 only for interactive local signing, because Keychain access can block unattended releases" >&2
  exit 1
fi

APPCAST_SOURCE="$(
  /usr/bin/find "$SPARKLE_ARCHIVES" -maxdepth 1 -name '*.xml' -type f -print | /usr/bin/head -n 1
)"
if [[ -z "$APPCAST_SOURCE" ]]; then
  echo "error: generate_appcast did not produce an XML appcast" >&2
  exit 1
fi
/usr/bin/ditto --norsrc "$APPCAST_SOURCE" "$RELEASES_OUT/appcast.xml"

python3 - "$RELEASES_OUT/appcast.xml" "$RELEASE_NOTES_URL" "$VERSION" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
release_notes_url = sys.argv[2]
version = sys.argv[3]
xml = path.read_text(encoding="utf-8")
replacements = [
    f"https://mtplx.com/releases/MTPLX-{version}.html",
    f"https://mtplx.com/releases/notes/MTPLX-{version}.html",
]
for old in replacements:
    xml = xml.replace(old, release_notes_url)
if release_notes_url not in xml:
    raise SystemExit("error: appcast does not contain the expected release notes URL")
path.write_text(xml, encoding="utf-8")
PY

if ! /usr/bin/grep -q 'sparkle:edSignature' "$RELEASES_OUT/appcast.xml"; then
  echo "error: appcast is missing Sparkle EdDSA signature" >&2
  exit 1
fi

cat > "$SITE_OUT/download.html" <<HTML
<!doctype html>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=$DMG_URL">
<title>Download MTPLX</title>
<p><a href="$DMG_URL">Download MTPLX $VERSION</a></p>
HTML

cat <<SUMMARY

Release artifact staged locally.
DMG: $DMG
SHA256: $DMG_SHA256
Size: $DMG_SIZE
Website payload: $SITE_OUT

No upload has been performed by this script.
SUMMARY
