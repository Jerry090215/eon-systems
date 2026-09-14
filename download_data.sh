#!/bin/bash
# ============================================================
# Eon Systems - Data Download Script
# ============================================================
# This script downloads the FlyWire connectome data required
# to run the embodied fly simulation.
#
# Total download size: ~850 MB
# Required disk space: ~2 GB (including extracted files)
#
# Data sources:
#   - FlyWire v783 connectome (Zenodo / codex.flywire.ai)
#   - Neuron annotations (FlyWire community)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/raw_data"
MUSHROOM_DIR="${SCRIPT_DIR}/mushroom_body"
VISUAL_DIR="${SCRIPT_DIR}/visual_lobe"

echo "============================================================"
echo "  Eon Systems - FlyWire Connectome Data Download"
echo "============================================================"
echo ""
echo "Data directory: ${DATA_DIR}"
echo ""

# Create directories
mkdir -p "${DATA_DIR}"
mkdir -p "${MUSHROOM_DIR}"
mkdir -p "${VISUAL_DIR}"

# -----------------------------------------------------------
# 1. FlyWire v783 proofread neuron IDs
# -----------------------------------------------------------
echo "[1/4] Downloading proofread neuron IDs (1.1 MB)..."
if [ ! -f "${DATA_DIR}/proofread_root_ids_783.npy" ]; then
    curl -L -o "${DATA_DIR}/proofread_root_ids_783.npy" \
        "https://zenodo.org/records/10824808/files/proofread_root_ids_783.npy"
    echo "  ✓ Downloaded"
else
    echo "  ✓ Already exists, skipping"
fi

# -----------------------------------------------------------
# 2. FlyWire v783 proofread connections (LARGE: 813 MB)
# -----------------------------------------------------------
echo ""
echo "[2/4] Downloading proofread connections (813 MB)..."
echo "      This is the largest file. It may take several minutes."
if [ ! -f "${DATA_DIR}/proofread_connections_783.feather" ]; then
    curl -L -o "${DATA_DIR}/proofread_connections_783.feather" \
        "https://zenodo.org/records/10824808/files/proofread_connections_783.feather"
    echo "  ✓ Downloaded"
else
    echo "  ✓ Already exists, skipping"
fi

# -----------------------------------------------------------
# 3. Neuron annotations (30 MB)
# -----------------------------------------------------------
echo ""
echo "[3/4] Downloading neuron annotations (30 MB)..."
if [ ! -f "${DATA_DIR}/neuron_annotations.tsv" ]; then
    curl -L -o "${DATA_DIR}/neuron_annotations.tsv" \
        "https://raw.githubusercontent.com/htem/FlyWire_annotations/main/Supplemental_file1_neuron_annotations.tsv"
    echo "  ✓ Downloaded"
else
    echo "  ✓ Already exists, skipping"
fi

# -----------------------------------------------------------
# 4. Pre-extracted mushroom body circuit (56 MB)
# -----------------------------------------------------------
echo ""
echo "[4/4] Downloading pre-extracted mushroom body circuit (56 MB)..."
echo "      (Alternatively, run scripts/extract_mushroom_body.py to extract from raw data)"
if [ ! -f "${MUSHROOM_DIR}/circuit.json" ]; then
    echo "  Note: Pre-extracted circuit is hosted separately."
    echo "  Please check the project README for the latest download link,"
    echo "  or run the extraction script:"
    echo "    python3 scripts/extract_mushroom_body.py"
    echo ""
    echo "  Creating placeholder..."
    echo '{"note": "Run scripts/extract_mushroom_body.py to generate this file"}' > "${MUSHROOM_DIR}/circuit.json"
else
    echo "  ✓ Already exists, skipping"
fi

# -----------------------------------------------------------
# Summary
# -----------------------------------------------------------
echo ""
echo "============================================================"
echo "  Download Complete!"
echo "============================================================"
echo ""
echo "Data files:"
ls -lh "${DATA_DIR}/" 2>/dev/null | grep -v "^total" || echo "  (empty)"
echo ""
echo "Next steps:"
echo "  1. Extract mushroom body circuit (if not pre-downloaded):"
echo "     python3 scripts/extract_mushroom_body.py"
echo ""
echo "  2. Extract visual lobe circuit:"
echo "     python3 scripts/extract_visual_lobe.py"
echo ""
echo "  3. Run the embodied fly simulation:"
echo "     cd embodied_fly && python3 embodied_fly_live.py"
echo ""
echo "============================================================"
