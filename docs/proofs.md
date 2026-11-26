# The proofs, in plain English

Each section gives a statement, the reason it is true, and the name of its Lean proof in
`lean/ThoroughMastication/Mastication.lean`. The Lean file builds on two files of the
[squiggly line library](https://github.com/Ghost---Shadow/squiggly-line-theorem):
`LeastAction.lean` and `Forgetting.lean`.

## The setting

The model is linear in its weights and the loss is the squared error. A batch is a map
`predict` from weights to the outputs on that batch, and a list of labels. Weights **fit** the
batch when `predict` sends them to the labels.

Some words used below:

- A **move along the features** of a batch is a move of the form `predict† vector`: a sum of
  the batch's feature vectors. Every gradient step on the batch is such a move.
- A direction is **unseen** by a batch when moving along it changes no output on the batch.
- A **mastication** takes start weights to weights that fit the batch, by a move along its features.
- A **full fit** is one set of weights that fits every batch. It exists when the model has
  more weights than the data has examples and the labels have no noise.

What is assumed and not proved: that gradient descent on a batch arrives at a fit. In general
it only comes as close as you like. The library makes the same assumption.

## 0. Gradient descent to a fit is a mastication

**Statement.** If gradient descent on a batch arrives at a fit of the batch, the weights it
arrives at are a mastication of the start weights.

**Why.** The library proves that gradient descent moves only along the features
(`gradient_descent_stays_in_the_directions_of_the_features`). The fit is given.

**Lean:** `gradient_descent_to_a_fit_is_a_mastication`

## 1. A move along the features is at right angles to what the batch does not see

**Statement.** If a move is along the features of a batch, and a direction is unseen by the
batch, the move and the direction are at right angles.

**Why.** The inner product of `predict† vector` with a direction is the inner product of
`vector` with `predict` of that direction. For an unseen direction that is the inner product
with zero.

**Lean:** `a_move_along_the_features_is_at_right_angles_to_what_the_batch_does_not_see`

## 2. A mastication brings the weights closer to a full fit by exactly its travel (C1)

**Statement.** Take a mastication and a full fit. Then

    (distance to the full fit before)² = (distance after)² + (travel of the mastication)²

**Why.** After the mastication, the weights and the full fit both fit the batch. So the way from one
to the other is unseen by the batch. The mastication itself is a move along the features. By
section 1 the two are at right angles, and Pythagoras gives the equation.

**So:** a mastication never moves the weights away from a full fit.

**Lean:** `mastication_brings_weights_closer_by_exactly_its_travel`,
`mastication_never_moves_weights_away_from_a_full_fit`

## 3. No use of the batch gets nearer than the mastication (C2)

**Statement.** Take any other weights reached from the same start by a move along the
features of the batch. They need not fit the batch. They are no nearer to a full fit than the
masticated weights are.

One gradient step, several steps, a mastication stopped early and a step of any size are all moves
along the features. So none of them beats the mastication at getting near a full fit.

**Why.** The move from the masticated weights to the other weights is a difference of two moves
along the features, so it is along the features too. The way from the masticated weights to the
full fit is unseen by the batch. They are at right angles, so by Pythagoras the other weights
are farther from the full fit, by the length of that move.

**Lean:** `no_use_of_the_batch_gets_nearer_than_the_mastication`

## 4. The squared travels add up to the distance gained (C3)

**Statement.** Masticate one batch after another, any number of batches. Then

    sum of (travel of each mastication)² = (first distance to the full fit)² − (last distance)²

**So:** the sum is at most the first squared distance, however many mastications there are. The
travels have to get shorter.

**Why.** Add up the equation of section 2 over the mastications. Each "distance after" is the next
mastication's "distance before", so everything in the middle cancels.

**Lean:** `squared_travels_add_up_to_the_distance_gained`,
`total_squared_travel_is_at_most_the_first_squared_distance`

## 5. All the forgetting together is bounded

**Statement.** Take an old batch, and look at how much each mastication changes the errors on it.
Then

    sum of (change in the old errors)² ≤ (size of the old predictor)² × (first distance to the full fit)²

The right side does not depend on the number of mastications. So the forgetting of the old batch
cannot go on at the same size forever.

**Why.** The library proves that a move changes the old errors by at most the size of the old
predictor times the length of the move (`forgetting_is_at_most_predict_size_times_distance`).
Square it, add up over the mastications, and use section 4.

**Lean:** `total_squared_forgetting_is_bounded`

## 6. With one number, a mastication keeps nothing of the past (C4, the part that is proved)

**Statement.** Fit one number to a batch of labels. For any guess,

    loss of the guess = loss of the batch mean + (number of labels) × (guess − batch mean)²

**So:** the batch mean is the only best guess. A mastication to convergence ends there, whatever the
start was. Nothing learned from earlier batches is left.

**Why.** Write each error of the guess as the error of the mean plus the gap between the
guess and the mean. Square and add up. The cross terms add up to the gap times the sum of the
mean's errors, and the mean's errors add up to zero.

**Not proved here:** that the error of a batch mean stays at the noise of one batch while a
running mean improves with every batch. That is the standard fact about the variance of a
mean. It is checked with numbers in `scripts/check_claims.py`.

**Lean:** `loss_of_a_guess_is_loss_of_the_batch_mean_plus_the_miss`,
`batch_mean_is_the_best_single_number`

# A mastication as a probe of a dataset

The sections above ask whether masticating is a good way to train. These ask what a mastication tells
you about the data. By section 3 a mastication takes everything a batch has to give, so what is still
wrong afterwards is what the batch did not contain.

## 7. A batch that already fits needs no travel

**Statement.** If the weights already fit a batch, a mastication of that batch leaves them where they
are. A duplicate of a batch that was just masticated is such a batch.

**So:** the travel of a mastication measures how new its batch is. It is zero for a duplicate, and by
section 2 it is never more than the distance to a full fit.

**Why.** Before and after the mastication the weights fit the batch, so the move is unseen by the
batch. The move is also along the features. By section 1 it is at right angles to itself, and
only the zero move is.

**Lean:** `masticating_a_batch_that_already_fits_moves_nothing`,
`travel_of_a_mastication_is_at_most_the_distance_to_a_full_fit`

## 8. What the batch spans is learned with it

**Statement.** Take another batch whose outputs are always a combination of this batch's
outputs: each of its examples has features that are a combination of this batch's features.
Suppose a full fit of both exists. Then any weights that fit this batch fit the other batch
too. No step was taken on it.

**So:** after masticating this batch, a mastication of the other batch moves nothing.

**Why.** The other batch's outputs are the combination of this batch's outputs. This batch's
outputs are its labels, both for the weights and for the full fit. So the other batch's
outputs are the same for the weights and for the full fit, and the full fit gives its labels.

**Lean:** `what_the_batch_spans_is_learned_with_it`, `a_batch_that_is_spanned_needs_no_travel`

## 9. A batch with separate features is left alone

**Statement.** If no move along the features of this batch changes the outputs of another
batch, a mastication of this batch leaves the outputs of the other batch exactly where they were.

**Why.** A mastication is a move along the features of its batch, and such a move changes nothing
on the other batch by assumption. The library proves the same for gradient steps
(`separate_features_forget_nothing`).

**Lean:** `masticating_a_batch_with_separate_features_changes_nothing_elsewhere`

## 10. In between: the error elsewhere is at most the distance times what sticks out

**Statement.** Take one other example. Split its feature vector into a part that is a
combination of the batch's features and the rest, the part that sticks out. For weights that
fit the batch,

    error on that example ≤ (size of the part that sticks out) × (distance from the weights to a full fit)

Sections 8 and 9 are the two ends: nothing sticks out, or everything does.

**Why.** The weights and the full fit both fit the batch, so their difference is unseen by the
batch. The part of the feature vector that the batch spans sees no difference between them.
Only the part that sticks out does, and an inner product is at most the product of the two
lengths.

**Lean:** `error_elsewhere_is_at_most_distance_times_what_sticks_out`

# A chain of chapters: where errors compound

Everything above is for a model with fixed features. There, nothing learned from one batch
changes how learnable the next one is, so nothing compounds. The mastery-learning argument, that without
chapter 3 there is no chapter 4, needs a model in which a later chapter is built on the
earlier ones. The simplest one is a chain: the output after chapter `k` is the input times the
weights of chapters 1 to `k`, multiplied together. So chapter `k` uses everything before it,
as a school chapter does.

## 11. The signal for a new chapter is the product of the earlier ones

**Statement.** Change the weight of chapter `k` by some amount. The output after chapter `k`
changes by that amount times the product of the earlier weights times the input.

**So:** the earlier chapters set how much learning signal the new chapter gets.

**Why.** The output after chapter `k` is (product of the earlier weights) times (weight of
chapter `k`) times the input. The earlier weights are not touched by the change, so the
output is a straight line in the weight of chapter `k`, with that slope.

**Lean:** `signal_for_a_new_chapter_is_the_product_of_the_earlier_ones`

## 12. Chapters that are mastered pass the whole signal

**Statement.** If the product of the earlier weights is one, a change in the new chapter's
weight reaches the output in full.

**Why.** Section 11, with the product equal to one.

**Lean:** `mastered_chapters_pass_the_whole_signal`

## 13. Half-learned chapters compound

**Statement.** If each of the `k` earlier weights is at most `m` in size, their product is at
most `m` to the power `k`.

**So:** with every chapter learned to 90%, the signal after ten chapters is at most 35%, and
after thirty at most 4%. This is the compounding. A learner that starts with all weights
small has almost no signal for a late chapter, which is why the late chapter cannot be
learned first.

**Why.** By induction on the number of chapters: the size of a product is the product of the
sizes, and each new factor is at most `m`.

**Lean:** `signal_is_at_most_mastery_to_the_power_of_the_chapters`

**In the library.** The same product appears for deep networks in
`gradient_coming_back_is_at_most_the_same_product` (the gradient coming back is bounded by
the product of the layer sizes), and `balance_after_step` says how training shares a product
between its factors. They are related results; the three theorems here do not use them.

## 14. A later chapter can make up for the earlier ones, without repairing them

**Statement.** In the chain, unless the product of the earlier weights is zero, there is a
weight for the new chapter that makes its output right, whatever the earlier weights are. And
whatever weight the new chapter takes, the output after the earlier chapters stays what it
was.

**So:** a later chapter can be right while an earlier one is still wrong. This is what
time-boxing did in the chain tests: it reached the last chapter with the earlier ones left
wrong.

**Why.** The output after the new chapter is (product of the earlier weights) times (the new
weight). Take the new weight to be one over that product. The earlier outputs do not contain
the new weight at all.

**Lean:** `a_later_chapter_can_make_up_for_the_earlier_ones`,
`making_up_does_not_repair_the_earlier_chapters`,
`changing_a_chapter_leaves_the_earlier_product_alone`

# Two more facts behind the tests

## 15. Masticating flipped labels gives the opposite weights

**Statement.** Start at zero. If a mastication of a batch ends at some weights, a mastication
of the same batch with every label multiplied by minus one ends at minus those weights.

**So:** a batch with flipped labels teaches the opposite of what it should. This is why the
bad batch had a negative row in the cross-mastication table.

**Why.** Minus the weights give minus the outputs, which are the flipped labels. And minus a
move along the features is a move along the features.

**Lean:** `masticating_flipped_labels_gives_the_opposite_weights`

## 16. A batch that sees everything pins the weights down

**Statement.** If no direction is unseen by a batch, any weights that fit it are the full fit.

**So:** with superset chapters, the last chapter sees every concept. Fitting it is fitting
everything, which is why mastery in order ended with every chapter learned, and why the
travels added up to the whole distance (section 4, with nothing left at the end).

**Why.** Two sets of weights that fit the batch differ by a direction the batch does not see,
and there is none but zero.

**Lean:** `weights_that_fit_a_batch_that_sees_everything_are_the_full_fit`

# What is proved and what is only measured

| Finding of a test | In Lean? |
|---|---|
| Duplicates and covered batches have an entry of 1 (Q1, Q2, S1) | yes, section 8 |
| Batches with separate features have an entry of 0 (Q3, Q7, S7) | yes, section 9 |
| A duplicate needs no travel (Q6) | yes, section 7 |
| A batch with flipped labels teaches the opposite (Q5, M3) | the mechanism, section 15; the size of the negative entry is measured |
| Batches of one group share about a third (Q4); a chapter teaches `k / j` of a later one (S2) | no: these are averages over random data |
| Mastery in order of superset chapters forgets nothing (S4) and costs only the new material (S5) | yes, sections 4, 8 and 16 together; the statement about a whole pass is not written as one theorem |
| Time-boxing sometimes hurt chapter 1 in the linear model (S6) | no: measured, and my explanation is untested |
| The signal for a late chapter is tiny when the weights are small (behind K1) | the bound, section 13; that gradient descent then stays stuck for a given number of steps is measured |
| Time-boxing reaches the last chapter with earlier ones wrong (K3, T2) | that it is possible, section 14; that gradient descent does it is measured |
| Mastery never leaves a chapter it cannot finish in time (T1) | no: a count of steps in one run |
| The chain forgets earlier chapters (K4) | no; the library's `balance_after_step` is the related result |
| Everything about the network (`mlp`) | no: the theorems are for models that are linear in their weights, and for the chain |

## How this meets the squiggly line

The library's picture of training is travel: a model moves from its start to a fit, and what
it can do to answers elsewhere is bounded by how far it travels. A mastication is one leg of that
trip. Sections 2 to 4 say the legs are never wasted when a full fit exists, and section 5
says the damage to earlier batches is paid for out of the same fixed budget of travel.
