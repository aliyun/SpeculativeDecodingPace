# SpecPace

Hugging Face Transformers implementation accompanying
*SpecPace: Guiding Edge-Cloud Speculative Decoding with Committed Progress*.
CP-driven joint execution decisions select synchronous or streaming drafting
and verification length. CP-driven history matching supplies local drafts
from cloud-resident history, using target-verification feedback.

## Run

Start the edge first, then the cloud on the other machine. Replace `EDGE_HOST`
with the edge IP or hostname; TCP port 38101 must be reachable from the cloud.
Use a trusted network or VPN for this HTTP connection.

```bash
CUDA_VISIBLE_DEVICES=0 python3 engines/launch.py --engine hf --role edge \
  --model /models/draft --host 0.0.0.0 --out /tmp/edge

CUDA_VISIBLE_DEVICES=0 python3 engines/launch.py --engine hf --role cloud \
  --model /models/target --host 0.0.0.0 --edge-url http://EDGE_HOST:38101 --out /tmp/cloud
```

Models, datasets and measurement artifacts are supplied separately. Baseline
entrypoints and licenses are in [`baselines/`](baselines/README.md).
Distribute source archives without Git metadata.
