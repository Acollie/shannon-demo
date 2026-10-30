# Server energy (lean): method

This is how the "Server energy (lean)" note on the GLM page was measured. It is a **server** measurement on the full
Wikidata graph, not a measurement of this page in your browser. Nothing on the page claims anything about energy on a
phone or laptop.

## The claim (reviewed wording)

> On one A40, answering 2,000 open Mintaka questions in a batch, Shannon's whole answer path (name lookup, graph walk,
> scoring; 23M parameters, 0.1M trained) used about 0.12 J per question above the GPU's idle draw (GPU measured, CPU
> estimated from TDP). Qwen2.5-1.5B closed-book used 0.25 J and Qwen2.5-0.5B 0.14 J, measured the same way on the same
> pod. Qwen-1.5B is still more accurate (27% vs 23%; Qwen is scored on its text, Shannon on graph nodes). Counting the
> GPU's idle draw as well, the three are 0.27, 0.40 and 0.26 J.

## Set-up

- One NVIDIA A40 pod (Runpod, EU-SE-1), 2026-10-25. Graph: Wikidata5M plus dates and quantities (43.3M edges, 5.04M
  nodes, 1,089 relations), the graph Shannon's KG16 experiments use. Questions: all 2,000 Mintaka **dev** questions,
  answered as one batch. Mintaka test was not read.
- Shannon: the KG16 R0 model (frozen MiniLM-L6 22.7M + about 0.1M trained parameters, 3 seeds). Accuracy 22.6% hits@1
  in Mintaka's KG mode (graph nodes).
- Qwen2.5-1.5B-Instruct and Qwen2.5-0.5B-Instruct, closed-book, vLLM, greedy, at most 32 new tokens. Accuracy 27.3% and
  17.0% in Mintaka's text mode (their words), which is more lenient than KG mode.

## What "lean" changed (same answers apart from the linker)

The answer path was re-engineered without retraining the scorer: an exact + label-prefix name index instead of a dense
search over 1M names (its 513-parameter linker MLP retrained on Mintaka train), no n-gram MiniLM encode, one shared
MiniLM pass for linker and reader, the graph on the GPU with each distinct start topic expanded once per batch, and all
candidates scored in batches. Checks: the GPU path listing equals the earlier CPU listing on 100% of questions, and the
batched scorer gives the same answer as the earlier scorer on 100% of questions. Accuracy 22.8% -> 22.6% (the linker
and fp16 MiniLM).

## Measurement

| System (2,000 Mintaka dev, batched, one A40) | J/q above idle (GPU + CPU) | J/q including GPU idle | Accuracy |
|---|---|---|---|
| Shannon lean | 0.124 (0.089 + 0.035) | 0.27 | 22.6% (KG mode) |
| Qwen2.5-1.5B closed-book | 0.247 (0.229 + 0.018) | 0.40 | 27.3% (text mode) |
| Qwen2.5-0.5B closed-book | 0.144 (0.125 + 0.019) | 0.26 | 17.0% (text mode) |

- **GPU**: NVML board energy counter, minus idle power (111.9 W, measured for 60 s with a CUDA context open) times wall
  time.
- **CPU**: not measured (RAPL is not readable in the pod). Estimated as process CPU-seconds (including worker processes)
  x 9.6 W per core (Xeon Gold 6342: 230 W TDP / 24 cores).
- **Left out**: one-off loading, DRAM and host power beyond the CPU estimate, network, cooling.

## Limits

- "Below Qwen-0.5B" holds only above idle; including the GPU's idle draw, Shannon lean (0.27 J) and Qwen-0.5B (0.26 J)
  are about the same.
- Batching helps Shannon: questions that share a start topic share listing work (2,797 distinct topics in 6,000 topic
  slots). With all-distinct topics the estimate is about 0.2 J per question (not measured). One question at a time
  would cost more for every system.
- Shannon is scored in KG mode and Qwen in text mode, so the accuracy columns are not like for like; Qwen-1.5B is more
  accurate either way.
- The lean path keeps a 3.5 GB name-vector table in RAM and the graph on the GPU. It is a server set-up; this page's
  browser engine is different (a 303k-entity slice of the graph, loaded in pieces, run on your CPU).
