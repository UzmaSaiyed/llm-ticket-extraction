# Baseline results

Original Qwen2.5-1.5B-Instruct, no fine-tuning. JSON validity was 1.0 in every run.

| Prompt | Split | Exact match | Intent | Category | Field F1 |
|---|---|---|---|---|---|
| Simple | validation | 0.000 | 0.000 | 0.000 | 0.000 |
| Simple | test | 0.000 | 0.000 | 0.000 | 0.000 |
| Engineered | validation | 0.500 | 0.634 | 0.724 | 0.679 |
| Engineered | test | 0.512 | 0.658 | 0.720 | 0.689 |
| Engineered with hints (rejected) | validation | 0.463 | 0.568 | 0.770 | 0.669 |

The engineered prompt lists all allowed intents and gives five worked examples.
Adding hints for similar intents lowered intent accuracy, so that version was reverted.