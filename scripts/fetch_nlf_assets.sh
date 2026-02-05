#!/usr/bin/env bash
set -euo pipefail

OWNER="isarandi"
REPO="nlf"

TAG_S="v0.2.2"
ASSET_S="nlf_s_multi_0.2.2.torchscript"

TAG_L="v0.3.2"
ASSET_L="nlf_l_multi_0.3.2.torchscript"

# These two are small, but if you want them fetched the same way:
ASSET_SMPLX="smplx.npy"
ASSET_SMPL="smpl.npy"

OUT_DIR="${1:-weights/nlf}"
mkdir -p "${OUT_DIR}"

get_url () {
  local tag="$1" asset="$2"
  python3 - <<PY
import json, os, sys, urllib.request
owner="${OWNER}"; repo="${REPO}"; tag="${tag}"; asset="${asset}"
api=f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
headers={"Accept":"application/vnd.github+json","User-Agent":"fetch-assets"}
tok=os.environ.get("GITHUB_TOKEN")
if tok: headers["Authorization"]=f"Bearer {tok}"
req=urllib.request.Request(api, headers=headers)
data=json.load(urllib.request.urlopen(req))
for a in data.get("assets", []):
    if a.get("name")==asset:
        print(a["browser_download_url"])
        raise SystemExit(0)
print("ERROR: asset not found:", asset, "in", tag, file=sys.stderr)
print("Available:", [a.get("name") for a in data.get("assets", [])], file=sys.stderr)
raise SystemExit(2)
PY
}

download () {
  local tag="$1" asset="$2" out="$3"
  local url
  url="$(get_url "$tag" "$asset")"
  echo "[DL] $asset"
  curl -L --retry 5 --retry-delay 2 -o "$out" "$url"
}

download "${TAG_S}" "${ASSET_S}" "${OUT_DIR}/${ASSET_S}"
download "${TAG_L}" "${ASSET_L}" "${OUT_DIR}/${ASSET_L}"
# optional:
# download "${TAG_S}" "${ASSET_SMPLX}" "${OUT_DIR}/${ASSET_SMPLX}"
# download "${TAG_S}" "${ASSET_SMPL}"  "${OUT_DIR}/${ASSET_SMPL}"
