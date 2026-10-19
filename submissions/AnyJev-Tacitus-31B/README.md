# AnyJev Tacitus · Gemma-4-31B

A decision model fine-tuned from [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it):
[morriszjm/AnyJev-Tacitus-31B](https://huggingface.co/morriszjm/AnyJev-Tacitus-31B) (bf16 weights, FP8 at load), read
through [anyjev](https://pypi.org/project/anyjev/) 0.3.0's restricted label readout. Its training and inference were
designed together; the method will be described in a forthcoming paper.

- Each question is rendered with AnyJev's prompt and read as the next-token distribution restricted to the option
  labels; nothing is generated.
- The options are read in two orders (the one given and its half rotation) and averaged in log space; no label prior.
- The model's full-attention layers read each prompt in two bidirectional blocks, the state and the question with its
  options. The engine sends both token ranges with every request; a vLLM plugin in the code repository applies them,
  so one prefill of a state serves every question and option order of its request.
- More than 26 options: each option gets a letter and a digit (A0..A9, B0..B9, ...); p(option) = p(letter) · p(digit | letter).
- Yes/no questions are shown with their criteria.

| | |
|---|---|
| Decision Index 0.3 public index | 65.52 (raw 73.50, breadth 64.80) |
| Areas | knowledge 52.8 · language 70.3 · retrieval 67.5 · tools 81.3 · arts 53.2 |
| Requests | 140,178 of 140,178 scoreable, all answered, none unsupported, no errors |
| Request time in `scores.json` | median 632 ms, mean 930 ms, measured with several client processes in parallel per server, so it includes queueing |
| Single-process latency | our stratified 750-row sample, one request at a time, `serve.sh` on one RTX PRO 6000 Blackwell (FP8): median 98 ms, mean 536 ms, p80 953 ms, p95 2,240 ms. The same sample and box give AnyJev L0 · Gemma-4-31B (on the board) 102 / 516 / 946 / 2,173 ms: this sample is heavier than the maintainers' |

## Run

Code, settings and the exact commands: [MorrisZJ/anyjev-tacitus](https://github.com/MorrisZJ/anyjev-tacitus/tree/3b21760d8573dac34aed21ca43cfd316c0ed0daf).

```sh
pip install -r requirements.txt && pip install ./serving     # vllm 0.29.0, anyjev 0.3.0, the plugin; plus this kit at 9eb2dbe
bash serve.sh                                                # ANYJEV_BIDI_GLOBAL=1 VLLM_USE_V2_MODEL_RUNNER=0 vllm serve morriszjm/AnyJev-Tacitus-31B --quantization fp8 ...
bash run.sh                                                  # level=L0 max_rotations=2 big_rotations=2 bidi=block prior=none
```

Results: [morriszjm/decision-index-results](https://huggingface.co/datasets/morriszjm/decision-index-results/tree/d82c491a187b0f015b589ba813084fb695d98b78/runs/anyjev-tacitus-31b-v1)
(written with `--compact`). The run was split over group-whole shards and two identical FP8 servers (tensor parallel
2, window 114,688); every shard used the settings above. The servers read a local copy of the model repository, byte
for byte the same files (the repository's LFS sha256 sums).
