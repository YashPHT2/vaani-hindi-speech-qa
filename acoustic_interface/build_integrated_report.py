"""Integrate the frozen architecture review and measured final QA results."""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def percent(value):
    return f"{100 * value:.1f}%"


def bounds(item, scale=1):
    if item["estimate"] is None:
        return "undefined"
    return f"{scale * item['estimate']:.2f} [{scale * item['lower']:.2f}, {scale * item['upper']:.2f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown-only", action="store_true")
    args = parser.parse_args()
    artifacts = HERE / "artifacts/transcript-final"
    analysis = json.loads((artifacts / "analysis.json").read_bytes())
    raw = json.loads((artifacts / "result.json").read_bytes())
    cap_audit = json.loads((artifacts / "asr-cap-audit.json").read_bytes())
    assert analysis["semantic_status"] == "model_primary_plus_independent_subset"
    old = PROJECT / "historical/report-v2.md"
    if not old.exists():
        old = PROJECT / "output/repository-final/report.md"
    architecture = old.read_text(encoding="utf-8").split("## A Hindi listening-comprehension cascade")[0]
    architecture = architecture.replace("# What actually delays a spoken reply?", "## Task 1. What delays a spoken reply?", 1)
    groups = analysis["groups"]
    ci = analysis["bootstrap"]["intervals"]
    names = {"clean": "Clean", "noise_10db": "10 dB noise", "all": "Both views"}
    results = table(["Condition", "WER", "CER", "Exact answers", "Supported correct", "Unsupported claims"],
                    [[names[key], percent(groups[key]["wer"]), percent(groups[key]["cer"]),
                      f"{groups[key]['exact_matches']}/{groups[key]['n']}",
                      f"{groups[key]['correct_count']}/{groups[key]['n']}",
                      f"{groups[key]['hallucinated_count']}/{groups[key]['n']}"] for key in names])
    errors = table(["Condition", "Word edits / reference words", "Character edits / reference characters"],
                   [[names[key], f"{groups[key]['word_edits']} / {groups[key]['reference_words']}",
                     f"{groups[key]['character_edits']} / {groups[key]['reference_characters']}"] for key in names])
    timing = table(["Condition", "Median reply seconds", "p95 seconds", "Pipeline RTF", "Timed views"],
                   [[names[key], f"{groups[key]['p50_completed_text_s']:.3f}", f"{groups[key]['p95_completed_text_s']:.3f}",
                     f"{groups[key]['pipeline_rtf']:.3f}", f"{groups[key]['timing_rows']}/{groups[key]['n']}"] for key in names])
    metrics = [("WER, %", "wer", 100), ("CER, %", "cer", 100), ("Exact answers, %", "answer_em", 100),
               ("Supported correct, %", "correct_rate", 100), ("Unsupported claims, %", "hallucinated_rate", 100),
               ("Bare abstentions, %", "abstention_rate", 100), ("Median reply, s", "p50_completed_text_s", 1),
               ("Pipeline RTF", "pipeline_rtf", 1)]
    intervals = table(["Metric", "Clean, 95% interval", "Noise, 95% interval", "Noise minus clean"],
                      [[name, bounds(ci["clean"][key], scale), bounds(ci["noise_10db"][key], scale),
                        bounds(ci["noise_minus_clean"][key], scale)] for name, key, scale in metrics])
    subgroups = table(["Condition and QA type", "Supported correct", "Unsupported claims", "Bare abstentions"],
                      [[names[key] + (", answerable" if answerable == "True" else ", unanswerable"),
                        f"{stats['correct_count']}/{stats['n']}", f"{stats['hallucinated_count']}/{stats['n']}",
                        f"{stats['abstained']}/{stats['n']}"] for key in ("clean", "noise_10db")
                       for answerable, stats in analysis["answerability"][key].items()])
    all_stats = groups["all"]
    primary = analysis["primary_only"]["all"]
    strict = analysis["strict"]["all"]
    software = raw["software"]
    text = f"""# Speech architectures and Hindi speech-grounded QA

Anonymous technical assignment for Vaani Research Lab. The submission combines a comparative speech-model architecture review with an implemented acoustic-memory investigation and a frozen final evaluation of a transcript cascade. The final evaluation covers 100 new source recordings in two acoustic conditions. The tested adapter interventions failed their generated-answer criteria. The final candidate was selected under that stopping rule before final inference.

{architecture}

## Task 2. System, data and evaluation

The deployed research task takes a Hindi speech recording and a separately supplied written Hindi question about its content. Frozen Whisper-small generates a recognized transcript. Frozen Qwen2.5-1.5B-Instruct generates a short text answer using that transcript and question. Thus the implemented system performs speech-grounded question answering. It has a written-question input and an offline clip boundary. Streaming speech generation and conversational turn-taking belong to the Task 1 analysis.

The final sample contains 100 FLEURS Hindi recordings excluded by source ID, normalized publisher text and canonical audio hash from the 112 historical recordings. Every source has a clean view and a deterministic 10 dB additive white-noise view. The conditions share gain and question. There are 200 views and 100 recording units. The longest recording is 24.48 seconds. Each condition includes 80 answerable and 20 unanswerable questions. The source selection, question set, waveform policy, model revisions, prompt, stopping limits, review subset and scorer were frozen before inference. The questions were model-authored and independently model-reviewed against publisher text. Fresh-test human listening was absent, and speaker independence is unknown. Fresh denotes the project split. Foundation-model pretraining overlap was not audited, and supplied questions can cue previously known facts, so a correct answer alone does not prove reliance on the recording.

Whisper-small is pinned to revision 973afd24965f72e36ca33b3055d56a652f456b4d. Hindi transcribe decoding is greedy with a 224-new-token ceiling. Qwen2.5-1.5B-Instruct is pinned to 989aa7980e4cf806f80c7fef2b1adb7bc71aa306. It uses the existing Hindi evidence-only instruction, literal recognized transcript, genuine question, zero acoustic positions and a greedy 48-new-token ceiling. Neither reference transcript nor answer enters generation. The final candidate executes ordinary ASR and Qwen inference, without DTW, posterior extraction, adapter or teacher overhead. All inputs and prompt embeddings are hash-bound.

WER uses corpus word edit totals. CER uses corpus code-point edits after removing normalized spaces. Normalization casefolds, applies Unicode NFC, maps punctuation to spaces, converts Unicode decimal digits to ASCII and collapses whitespace, preserving Hindi marks. Empty or failed ASR counts as deletion of its full reference. Completed ASR is preserved if QA fails. All 200 responses remain in answer denominators, including capped or failed responses. WER can exceed 100% through insertions.

Supported correctness requires a complete answer justified by publisher evidence and no unsupported addition. An unsupported-claim response contains at least one invented or contradicted factual assertion. Faithful but irrelevant text can be incorrect without hallucinating. Bare अज्ञात or unknown is an abstention and is correct only for an unanswerable question. The model primary reviewer rated all {analysis['unique_review_cases']} distinct source/question/answer cases with condition names hidden. The independent reviewer rated {analysis['secondary_review_cases']} distinct cases covering the prospectively fixed 20 sources, including 16 answerable and four unanswerable sources. The other 80 sources have primary review only. Both original judgments and rationales are preserved. Correctness or hallucination conflicts become incorrect and ambiguous, with the hallucination flag combined by logical OR. Strict sensitivity additionally rejects all ambiguous answers. These are model judgments against text, rather than human listening judgments.

## Final held-out results

{results}

{errors}

The pooled supported-correct result is {all_stats['correct_count']}/200 ({percent(all_stats['correct_rate'])}), with {all_stats['hallucinated_count']}/200 unsupported-claim judgments. The primary-only count was {primary['correct_count']}/200 and the strict count is {strict['correct_count']}/200. The two reviewers disagreed on correctness or unsupported claims for {len(analysis['conflicts'])} distinct cases in the reviewed subset. The full raw outputs, originals and effective labels are available for audit. These results describe the frozen baseline. Adapter efficacy is evaluated on its separate historical development recordings below.

{subgroups}

The run retained {all_stats['failed']} technical failures, {all_stats['empty']} empty answers and {all_stats['truncated']} responses ending at the QA token ceiling. Actual capped text was judged. Bare abstentions occurred {all_stats['abstained']}/200 times. On answerable views, the model abstained {analysis['answerability']['clean']['True']['abstained']}/80 times in clean audio and {analysis['answerability']['noise_10db']['True']['abstained']}/80 times under noise.

An independent audit corrected an ASR metadata error. The frozen worker subtracted the decoder prefix length from token arrays whose prefixes had already been removed by Transformers. This understated the raw cap count. The verified arrays contain {cap_audit['capped_views']} views with 224 returned tokens, including {cap_audit['by_condition']['clean']} clean and {cap_audit['by_condition']['noise_10db']} noisy views. The derived count leaves the original outputs, WER, CER, QA scores and timing intact. The separate audit script and JSON retain the calculation. [Pinned Whisper generation code]({cap_audit['source']})

## Timing and uncertainty

{timing}

Online timing starts with a prepared waveform in memory and includes ASR preprocessing, device transfer, recognition, transcript decoding, QA packing, tokenization, prefill, generation and final text decoding. CUDA is synchronized at the boundaries. Model loading, waveform construction and decoding, disk/network reads, output serialization, prompt-proof hashing and annotation are excluded. ASR accounts for {percent(all_stats['asr_rtf'] / all_stats['pipeline_rtf'])} of summed online time. The final run used {raw['gpu']} and peak allocated CUDA memory {raw['peak_cuda_bytes'] / 1024**3:.3f} GiB. Model loading took {raw['model_loading_s']:.3f} seconds and the function took {raw['elapsed_s']:.3f} seconds including validation and artifact handling. There were zero warmup views, so the first timed view includes online kernel initialization. RTF divides summed online processing time by the matching audio duration. It measures offline processing speed on clips, without microphone endpoint or playback delay.

{intervals}

Brackets contain percentile 95% source-cluster bootstrap intervals from 2,000 draws with NumPy default_rng(17). Each draw samples 100 source IDs from ascending decimal order with replacement and carries both conditions. Rates and RTF are recomputed from sampled totals. Noise-minus-clean percentage effects are percentage-point differences. The clean unsupported-claim interval is degenerate because the sample had zero such events. Population risk remains uncertain. The intervals condition on fixed annotations and exclude reviewer, speaker and dataset-selection uncertainty. Replicate counts are retained in analysis.json.

The measured inference software was Python {software['python']}, PyTorch {software['torch']}, CUDA {software['cuda']}, Transformers {software['transformers']} and NumPy {software['numpy']}. Offline bootstrap analysis used Python 3.12 and NumPy 2.5.1. The exact image, code, revisions, resource ceilings and file hashes accompany the outputs.

## Implemented architectural extension and ablations

The research hypothesis was that a compact learned acoustic interface could recover answer-relevant evidence omitted by an ASR transcript. The implementation feeds frozen Whisper encoder features to a 305,281-parameter question-conditioned connector, selecting at most four continuous vectors for frozen Qwen alongside the recognized transcript. Predicted-prefix posterior statistics and availability flags provide uncertainty inputs. DTW-derived spans propose candidates. A reference-free 20-frame fallback supplies candidates when the ASR alignment is empty. The query-conditioned selector, pooling, projection and gates are learned, with the declared straight-through training estimator. These operations are implemented and were exercised with real pretrained models and matched training.

Learned speech-to-language projection, query-based speech pooling and response-distribution alignment have direct precedents in SALMONN, BLSP, Wav2Prompt, SLAM-ASR and BLSP-KD. This project investigates their limited adaptation to a four-vector, query- and uncertainty-conditioned interface. It does not claim to invent those broader mechanisms. The detailed prior-art comparison is retained in acoustic_interface/sources.md. [SALMONN](https://arxiv.org/abs/2310.13289), [BLSP](https://arxiv.org/abs/2309.00916), [Wav2Prompt](https://arxiv.org/abs/2406.00522), [SLAM-ASR](https://arxiv.org/abs/2402.08846), [BLSP-KD](https://arxiv.org/abs/2405.19041)

The investigation separates three bottlenecks. First, candidate construction can fail before the language model receives acoustic evidence. The archived pilot had 12 empty views among 88 training/development views. On selected noisy source 1806, the fallback restored 35 candidates and four memory slots, but all tested readouts still answered incorrectly. Second, acoustic influence can occur without useful readout. Frozen-decoder omission and masking tests showed likelihood sensitivity in one selected recording, while the first shared-prefix token localization contrast was negative. Human timing was approximate and full phrase extent remained uncertain. Actual and donor audio often produced comparable QA utility. Third, response generation can fail despite correct text. The matched publisher-transcript development control reached 14/24 correct versus 6/24 for recognized text, leaving 10 incorrect views even with reference text. These are separate observations with separate denominators.

Global magnitude calibration tested one scalar on the fixed memory. The sensitivity preflight passed, but 32 CE updates fitted alpha to 0.981649 and changed none of the 22 primary actual-audio answers or their matched donor answers. The registered utility rule failed. This rejects the tested scalar schedule while leaving broader representation questions open.

The final adapter intervention changed only the training objective. CE and CE+KL each independently initialized from the same pilot checkpoint and received identical four-epoch schedules on 32 historical training sources and 64 views, with 32 updates per arm. Only connector parameters trained. A training-only frozen Qwen teacher received publisher text. Eleven answerable and three unanswerable sources passed the frozen exact-match eligibility test, covering 28/64 views and 250/628 supervised answer tokens. All views retained CE. Eligible views added 0.5 times temperature-squared teacher-to-student KL at temperature two, aligned at identical answer-token positions including EOS. The final endpoint was fixed before development scoring.

| Arm on 11 historical development sources, 22 views | Supported correct | Strict correct | Mean answer NLL |
| --- | --- | --- | --- |
| Transcript only | 6/22 | 4/22 | 1.724276 |
| Matched CE, actual audio | 3/22 | 2/22 | 1.496684 |
| Matched CE, donor audio | 5/22 | 4/22 | 1.462607 |
| CE+KL, actual audio | 4/22 | 4/22 | 1.451039 |
| CE+KL, donor audio | 5/22 | 4/22 | 1.476569 |

The CE+KL actual arm had 7/22 unsupported-claim judgments, versus 4/22 for CE actual and 2/22 for transcript only. The mean NLL reduction did not pass the prospectively required generated-answer rule. It required two distinct unambiguous rescues depending on matching audio, source-balanced gains over CE and donor changes, and at most one regressed source, under both rating policies. Only one source supplied the required specific rescue. In source 1949, CE invented an insect count of 12, while CE+KL actual correctly abstained and CE+KL donor still answered 12. Transcript only already abstained correctly. On source 1952, CE+KL changed a correct abstention into an unsupported mountain-pass elevation of 1500. Thus the strongest rescue restored abstention and supplied no demonstrated recovery of a spoken factual answer.

Reconciled actual gain and gain minus donor change each totaled 0.5 source-equivalents, below the required 1.0. Strict scoring reached those aggregate thresholds but still failed the two-source unambiguous rescue clause. Thirty-seven new distinct outputs received two blinded model reviews. The independent audit verified hashes, matched update schedules, finite losses/gradients, frozen buffers and local connector tensor changes. Full backbone and teacher invariance relies on cloud assertions and reviewed code. A one-frame duration-rounding caveat affected descriptive training metadata on source 1788, without consuming extra fallback or donor frames. The frozen implementation and caveat are retained.

The investigated interface exposes frozen language-model predictions to speech representations, but the tested objectives did not reliably convert that influence into supported factual-answer recovery. The conclusion concerns this initialization, data, four-vector budget and schedules. Historical development sources had prior exposure, views are correlated, speaker overlap is unknown, teacher coverage is incomplete and answer review is model-based. The results do not resolve all possible compact interfaces or the information content of the speech encoder.

## Historical system comparison and submission scope

The earlier 100-recording study compared automatic-language Whisper, a Hindi-language hint and publisher-text control on H200 MIG hardware. Supported correctness was 52/200, 56/200 and 144/200. The hint's +2 percentage-point gain had a source-bootstrap interval spanning -1.0 to +5.5 points and failed the registered noisy-latency guardrail. This is an implemented baseline/intervention comparison with a negative overall deployment decision. Those original predictions and reviews accompany the package. Its recordings later became historical research data and are disjoint from the fresh final 100. A10 and H200 times, different prompts and different recording sets support no direct system ranking across studies.

The assignment's improvement requirement is addressed through implemented and evaluated interventions. The fallback improved access to candidate features and KL improved a likelihood measure. Robust generated-answer improvement was not demonstrated. The final report preserves that outcome. Architecture optimization stopped under the registered rule, and the transcript-only candidate was frozen before this final inference. The final 100-source results triggered no additional tuning or repeat experiments.

The package includes code, editable diagrams, pretrained checkpoint revisions, frozen inputs, raw outputs, original and effective ratings, source-paired scoring, model/resource metadata and checksums. Run verify_submission.py, then the offline analysis command in README.md to reproduce the final metrics. Report.md is the editable source for the supplied anonymous PDF. Publication to GitHub and email submission remain separate user-controlled actions.
"""
    source = HERE / "submission_report.md"
    source.write_text(text, encoding="utf-8")
    if not args.markdown_only:
        sys.path.insert(0, str(PROJECT))
        from build_report import build
        build(source, PROJECT / "output/pdf/vaani-final-research-assignment.pdf", False)
    print(source)


if __name__ == "__main__":
    main()
