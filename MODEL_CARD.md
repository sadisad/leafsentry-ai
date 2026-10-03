# LeafSentry model and system card

## Intended use

A portfolio reference for CPU image inference, input validation, selective prediction, and measurable evaluation. Intended users are engineers exploring the API and system design. It is not an agronomic diagnostic tool or treatment recommender.

Educational decision-support demo; this is not a field-validated diagnosis.

## Model and data provenance

| Artifact | Immutable identity | Declared license |
|---|---|---|
| Model | `nateraw/vit-base-beans` at `41f85ace09a4613c2c65495b3b8465c4ceee1d00` | Apache-2.0 |
| Dataset | `AI-Lab-Makerere/beans` at `27aa014ce09b193e1a6f58112d4a66e0eddb69c5` | MIT |
| Held-out test ZIP | SHA-256 `ca67b15d960d1e2fd9d23fd8498ce86818ead90755c630a43baf19ec4af09312` | See dataset card |

Sources: [pinned model card](https://huggingface.co/nateraw/vit-base-beans/blob/41f85ace09a4613c2c65495b3b8465c4ceee1d00/README.md), [pinned dataset card](https://huggingface.co/datasets/AI-Lab-Makerere/beans/blob/27aa014ce09b193e1a6f58112d4a66e0eddb69c5/README.md).

The Beans dataset has 1,034 training, 133 validation, and 128 test images. The three labels are angular leaf spot, bean rust, and healthy. The upstream model is a fine-tuned Vision Transformer. This project does not retrain it or claim model novelty. Its upstream card reports 94.53125% test accuracy; the repository's reproduced results are independent executions with their own software/runtime context, not a claimed improvement over the upstream benchmark.

## Inference and decision policy

The input guard checks encoded size, supported MIME/format agreement, complete decode, dimensions, pixel count, and brightness/contrast/sharpness heuristics. It does not recognize the crop or detect all distribution shift. EXIF orientation is normalized before RGB conversion.

The backend loads the pinned processor and model with `trust_remote_code=False` and an explicitly selected slow processor (`use_fast=False`). Raw logits feed temperature softmax, confidence, top-two margin, and normalized entropy. Temperature defaults to 1.0, so confidence remains uncalibrated. The code supports temperature adjustment but no calibration fit has been performed.

Default score policy: confidence >= 0.75 and margin >= 0.20. Both must pass. Poor quality returns an abstention without model scores. These thresholds are demonstration choices; the held-out sweep must not be treated as validation-selected deployment policy.

## Evaluation boundaries

[Evidence](docs/evaluation/README.md) contains a three-sample wiring smoke and a full 128-image model-level test evaluation. The latter bypasses the HTTP/image-quality service guard and reports score-policy coverage, not service coverage. The upstream test split is small and comes from the model's own benchmark domain; it is not an external, farm-, device-, or geography-stratified evaluation.

There is no trained OOD detector, independent disease ground truth, prospective field test, fitted calibration, fairness analysis, load/SLO benchmark, or pesticide safety assessment. High confidence cannot establish that an unsupported photo is valid. A healthy output cannot rule out disease outside the three-class label set.

## Failure costs and responsible use

A confident false healthy prediction could delay care; a false disease prediction could prompt unnecessary treatment. The service therefore avoids treatment advice and makes uncertainty, thresholds, and limitations visible. Abstention reduces forced predictions but does not guarantee safety. Use expert review for any agricultural decision.
