# The proofs, in plain English

Each section gives a statement, the reason it is true, and the name of its Lean proof in
`lean/ThoroughMastication/Chew.lean`. The Lean file builds on two files of the
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
- A **chew** takes start weights to weights that fit the batch, by a move along its features.
- A **full fit** is one set of weights that fits every batch. It exists when the model has
  more weights than the data has examples and the labels have no noise.

What is assumed and not proved: that gradient descent on a batch arrives at a fit. In general
it only comes as close as you like. The library makes the same assumption.

## 0. Gradient descent to a fit is a chew

**Statement.** If gradient descent on a batch arrives at a fit of the batch, the weights it
arrives at are a chew of the start weights.

**Why.** The library proves that gradient descent moves only along the features
(`gradient_descent_stays_in_the_directions_of_the_features`). The fit is given.

**Lean:** `gradient_descent_to_a_fit_is_a_chew`

## 1. A move along the features is at right angles to what the batch does not see

**Statement.** If a move is along the features of a batch, and a direction is unseen by the
batch, the move and the direction are at right angles.

**Why.** The inner product of `predict† vector` with a direction is the inner product of
`vector` with `predict` of that direction. For an unseen direction that is the inner product
with zero.

**Lean:** `a_move_along_the_features_is_at_right_angles_to_what_the_batch_does_not_see`

## 2. A chew brings the weights closer to a full fit by exactly its travel (C1)

**Statement.** Take a chew and a full fit. Then

    (distance to the full fit before)² = (distance after)² + (travel of the chew)²

**Why.** After the chew, the weights and the full fit both fit the batch. So the way from one
to the other is unseen by the batch. The chew itself is a move along the features. By
section 1 the two are at right angles, and Pythagoras gives the equation.

**So:** a chew never moves the weights away from a full fit.

**Lean:** `chew_brings_weights_closer_by_exactly_its_travel`,
`chew_never_moves_weights_away_from_a_full_fit`

## 3. No use of the batch gets nearer than the chew (C2)

**Statement.** Take any other weights reached from the same start by a move along the
features of the batch. They need not fit the batch. They are no nearer to a full fit than the
chewed weights are.

One gradient step, several steps, a chew stopped early and a step of any size are all moves
along the features. So none of them beats the chew at getting near a full fit.

**Why.** The move from the chewed weights to the other weights is a difference of two moves
along the features, so it is along the features too. The way from the chewed weights to the
full fit is unseen by the batch. They are at right angles, so by Pythagoras the other weights
are farther from the full fit, by the length of that move.

**Lean:** `no_use_of_the_batch_gets_nearer_than_the_chew`

## 4. The squared travels add up to the distance gained (C3)

**Statement.** Chew one batch after another, any number of batches. Then

    sum of (travel of each chew)² = (first distance to the full fit)² − (last distance)²

**So:** the sum is at most the first squared distance, however many chews there are. The
travels have to get shorter.

**Why.** Add up the equation of section 2 over the chews. Each "distance after" is the next
chew's "distance before", so everything in the middle cancels.

**Lean:** `squared_travels_add_up_to_the_distance_gained`,
`total_squared_travel_is_at_most_the_first_squared_distance`

## 5. All the forgetting together is bounded

**Statement.** Take an old batch, and look at how much each chew changes the errors on it.
Then

    sum of (change in the old errors)² ≤ (size of the old predictor)² × (first distance to the full fit)²

The right side does not depend on the number of chews. So the forgetting of the old batch
cannot go on at the same size forever.

**Why.** The library proves that a move changes the old errors by at most the size of the old
predictor times the length of the move (`forgetting_is_at_most_predict_size_times_distance`).
Square it, add up over the chews, and use section 4.

**Lean:** `total_squared_forgetting_is_bounded`

## 6. With one number, a chew keeps nothing of the past (C4, the part that is proved)

**Statement.** Fit one number to a batch of labels. For any guess,

    loss of the guess = loss of the batch mean + (number of labels) × (guess − batch mean)²

**So:** the batch mean is the only best guess. A chew to convergence ends there, whatever the
start was. Nothing learned from earlier batches is left.

**Why.** Write each error of the guess as the error of the mean plus the gap between the
guess and the mean. Square and add up. The cross terms add up to the gap times the sum of the
mean's errors, and the mean's errors add up to zero.

**Not proved here:** that the error of a batch mean stays at the noise of one batch while a
running mean improves with every batch. That is the standard fact about the variance of a
mean. It is checked with numbers in `scripts/check_claims.py`.

**Lean:** `loss_of_a_guess_is_loss_of_the_batch_mean_plus_the_miss`,
`batch_mean_is_the_best_single_number`

## How this meets the squiggly line

The library's picture of training is travel: a model moves from its start to a fit, and what
it can do to answers elsewhere is bounded by how far it travels. A chew is one leg of that
trip. Sections 2 to 4 say the legs are never wasted when a full fit exists, and section 5
says the damage to earlier batches is paid for out of the same fixed budget of travel.
