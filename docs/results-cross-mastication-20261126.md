# Results of the cross-mastication test, 26 Nov 2026

The test of [cross-mastication-plan.md](cross-mastication-plan.md) was run once, on the
processor, in 26 seconds. The predictions were committed before the code existed, in
[../experiments/PREDICTIONS-cross-mastication.md](../experiments/PREDICTIONS-cross-mastication.md).
Every number below is from `python experiments/cross_mastication.py --summarise`; its full
output, with the two 20 by 20 tables, is in `experiments/results/cross-mastication-summary.txt`.
All 200 mastications reached their target, in at most 130 steps.

## The verdicts

`entry(i, j)` is the share of batch `j`'s loss that masticating batch `i` removed. Each
prediction held in 5 of 5 seeds.

| | Prediction | Measured |
|---|---|---|
| Q1 | Duplicates (0 and 6): both entries at least 0.99 | 1.0000 both ways |
| Q2 | Batch 1 covers batch 7: at least 0.99 one way, at most 0.9 back | 1.0000 one way; 0.15 to 0.48 back |
| Q3 | Different groups: at most 0.001 in size | exactly 0 |
| Q4 | Same group: mean entry between 0.2 and 0.45 | 0.26 to 0.37 (the theory says 1/3) |
| Q5 | The bad batch has a mean entry of at most −0.5 on its group, the lowest of all | −0.59 to −1.25; the lowest other batch was −0.12 |
| Q6 | In one pass, a duplicate or covered batch travels at most 1% of its original | 0.03% to 0.25% |
| Q7 | A new group arrives untouched | exactly untouched |
| Q8 | Simple rules recover the three groups, the duplicate pair and the covering pair | recovered exactly |
| M1 | `mlp`: duplicates at least 0.9 both ways | 0.9999 |
| M2 | `mlp`: entries across groups at most half of those inside a group | 0.02 to 0.03 against 0.25 to 0.36 |
| M3 | `mlp`: the bad batch has the lowest mean entry on its group | −0.62 to −1.26; the lowest other was −0.14 |

## What this shows

- **The table reads the planted structure.** With rules fixed in advance it found the groups,
  the duplicate, the covered batch and the batch with wrong labels, in every seed.
- **The entries mean what the theorems say.** Duplicates and covered batches came out at 1,
  separate groups at exactly 0, and batches that share a group at about a third, which is the
  share of the group's directions that one batch spans.
- **Wrong labels show as a negative row.** Masticating the bad batch made its group mates
  worse by about as much as their whole loss.
- **Novelty shows as travel.** A batch that brings nothing new moved the weights by a
  thousandth of what its original did.

## What this does not show

- **It was an easy test.** Q1 to Q3, Q6 and Q7 are theorems of `docs/proofs.md` seen in numbers;
  they could only fail if the code were wrong. Q5 and Q8 are the real tests, and the data
  were clean, with one fault of each kind and labels without noise.
- **The margin for the bad batch was not always wide.** In one seed its mean entry was −0.59,
  against a threshold of −0.5.
- **The network result says less than it seems.** A wide network with small inputs, taken at
  its starting weights and moved only a little, behaves almost like a model that is linear in
  its weights. So M1 to M3 do not show that the table survives once a network has learned
  its own features.
- **Single entries are noisy.** Clean batches of one group had single entries from about
  −0.5 to 0.6. Only averages over a row were predicted and tested.
- **No real data.** Whether the table is useful on a real dataset, at a real size, is open.
  The cost there is one mastication per batch and one measurement per pair.
