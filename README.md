# PathSnap

Distributed map-matching: candidate classifier + HMM/Viterbi, served via FastAPI, UI on a second laptop.

## Setup
    pip install -r requirements.txt

## Step 1 - exploration
    python scripts/01_inventory.py --data-root "<path>/Student folder"
    python scripts/02_folium_map.py --data-root "<path>/Student folder" --split train --trip 00000005

(More sections are added as each stage is built.)
