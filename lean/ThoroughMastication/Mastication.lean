/-
Masticating a batch: training on one batch until it fits, then the next.

The plain English of every proof is in `docs/proofs.md`.

For a model linear in its weights:
* `gradient_descent_to_a_fit_is_a_mastication` : gradient descent that arrives at a fit of a batch
  is a mastication.
* `mastication_brings_weights_closer_by_exactly_its_travel` : the squared distance to a full fit
  falls by the squared travel of the mastication.
* `no_use_of_the_batch_gets_nearer_than_the_mastication` : no other move along the features of the
  batch ends nearer to a full fit.
* `squared_travels_add_up_to_the_distance_gained` : over many mastications, the squared travels add
  up to the squared distance gained.
* `total_squared_forgetting_is_bounded` : the changes in the errors of an old batch, squared
  and added up over all mastications, stay under a ceiling that does not grow with the mastications.

A mastication as a probe of what one batch teaches about another:
* `masticating_a_batch_that_already_fits_moves_nothing` : a duplicate of a batch already masticated
  needs no travel.
* `what_the_batch_spans_is_learned_with_it` : a batch whose outputs are a combination of
  this batch's outputs is fitted with it, with no step taken on it.
* `masticating_a_batch_with_separate_features_changes_nothing_elsewhere` : a batch that shares
  no features is left exactly alone.
* `error_elsewhere_is_at_most_distance_times_what_sticks_out` : in between, the error on
  another example is bounded by how far its features stick out of what the batch spans.

For one number fitted to a batch:
* `batch_mean_is_the_best_single_number` : the batch mean is the best guess, so a mastication to
  convergence keeps nothing of the start.
-/
import SquigglyLine.Supporting.Training.LeastAction
import SquigglyLine.Supporting.NewInputs.Forgetting

namespace ThoroughMastication

open SquigglyLine

/-! ### A mastication, and what it does to the distance to a full fit -/

section Mastication

open ContinuousLinearMap

variable {Weights Outputs OldOutputs : Type*}
  [NormedAddCommGroup Weights] [InnerProductSpace ℝ Weights]
  [CompleteSpace Weights]
  [NormedAddCommGroup Outputs] [InnerProductSpace ℝ Outputs]
  [CompleteSpace Outputs]
  [NormedAddCommGroup OldOutputs] [InnerProductSpace ℝ OldOutputs]
  [CompleteSpace OldOutputs]

/-- A mastication of one batch: `masticated_weights` fit the batch and are reached from
`start_weights` by a move along the features of the batch. -/
def IS_A_MASTICATION
    (predict : Weights →L[ℝ] Outputs)
    (labels : Outputs)
    (start_weights : Weights)
    (masticated_weights : Weights) : Prop :=
  predict masticated_weights = labels ∧
    ∃ vector : Outputs, masticated_weights = start_weights + adjoint predict vector

/-- **Gradient descent to a fit is a mastication.** If gradient descent on a batch arrives at a fit
of the batch, the weights it arrives at are a mastication of the start weights. -/
theorem gradient_descent_to_a_fit_is_a_mastication
    (predict : Weights →L[ℝ] Outputs)
    (labels : Outputs)
    (learning_rate : ℝ)
    (start_weights : Weights)
    (step_count : ℕ)
    (trained_weights_fit :
      predict ((TAKE_GRADIENT_STEP predict labels learning_rate)^[step_count]
        start_weights) = labels) :
    let trained_weights :=
      (TAKE_GRADIENT_STEP predict labels learning_rate)^[step_count]
        start_weights
    IS_A_MASTICATION predict labels start_weights trained_weights := by
  unfold_named_pieces
  exact ⟨trained_weights_fit,
    gradient_descent_stays_in_the_directions_of_the_features predict labels
      learning_rate start_weights step_count⟩

/-- **A move along the features is at right angles to what the batch does not see.** -/
theorem a_move_along_the_features_is_at_right_angles_to_what_the_batch_does_not_see
    (predict : Weights →L[ℝ] Outputs)
    {start_weights : Weights}
    {moved_weights : Weights}
    {vector : Outputs}
    {direction : Weights}
    (weights_move_along_features :
      moved_weights = start_weights + adjoint predict vector)
    (direction_is_unseen : predict direction = 0) :
    inner ℝ (moved_weights - start_weights) direction = 0 := by
  rw [weights_move_along_features, add_sub_cancel_left, adjoint_inner_left,
    direction_is_unseen, inner_zero_right]

/-- **A mastication brings the weights closer to a full fit by exactly its travel.** The squared
distance to any weights that fit the batch falls by the squared travel of the mastication. -/
theorem mastication_brings_weights_closer_by_exactly_its_travel
    (predict : Weights →L[ℝ] Outputs)
    {labels : Outputs}
    {start_weights : Weights}
    {masticated_weights : Weights}
    {full_fit : Weights}
    (is_a_mastication : IS_A_MASTICATION predict labels start_weights masticated_weights)
    (full_fit_fits : predict full_fit = labels) :
    let distance_before := ‖start_weights - full_fit‖
    let distance_after  := ‖masticated_weights - full_fit‖
    let travel          := ‖masticated_weights - start_weights‖
    distance_before ^ 2 = distance_after ^ 2 + travel ^ 2 := by
  unfold_named_pieces
  obtain ⟨mastication_fits, vector, weights_move_along_features⟩ := is_a_mastication
  -- Both fit the batch, so the way between them is unseen by the batch.
  have rest_is_unseen : predict (masticated_weights - full_fit) = 0 := by
    rw [map_sub, mastication_fits, full_fit_fits, sub_self]
  have right_angle :
      inner ℝ (masticated_weights - full_fit) (masticated_weights - start_weights) = 0 := by
    rw [real_inner_comm]
    exact a_move_along_the_features_is_at_right_angles_to_what_the_batch_does_not_see
      predict weights_move_along_features rest_is_unseen
  have split : start_weights - full_fit
      = (masticated_weights - full_fit) - (masticated_weights - start_weights) := by
    abel
  -- Pythagoras.
  rw [split,
    norm_sub_sq_real (masticated_weights - full_fit) (masticated_weights - start_weights),
    right_angle]
  ring

/-- **A mastication never moves the weights away from a full fit.** -/
theorem mastication_never_moves_weights_away_from_a_full_fit
    (predict : Weights →L[ℝ] Outputs)
    {labels : Outputs}
    {start_weights : Weights}
    {masticated_weights : Weights}
    {full_fit : Weights}
    (is_a_mastication : IS_A_MASTICATION predict labels start_weights masticated_weights)
    (full_fit_fits : predict full_fit = labels) :
    let distance_before := ‖start_weights - full_fit‖
    let distance_after  := ‖masticated_weights - full_fit‖
    distance_after ≤ distance_before := by
  unfold_named_pieces
  have squares : ‖start_weights - full_fit‖ ^ 2
      = ‖masticated_weights - full_fit‖ ^ 2 + ‖masticated_weights - start_weights‖ ^ 2 :=
    mastication_brings_weights_closer_by_exactly_its_travel predict is_a_mastication full_fit_fits
  by_contra moved_away
  rw [not_le] at moved_away
  nlinarith [norm_nonneg (start_weights - full_fit),
    norm_nonneg (masticated_weights - full_fit),
    sq_nonneg ‖masticated_weights - start_weights‖]

/-- **No use of the batch gets nearer than the mastication.** Any other weights reached from the
same start by a move along the features of the batch, fitting it or not, are no nearer to a
full fit than the masticated weights. A single gradient step and a mastication stopped early are such
moves. -/
theorem no_use_of_the_batch_gets_nearer_than_the_mastication
    (predict : Weights →L[ℝ] Outputs)
    {labels : Outputs}
    {start_weights : Weights}
    {masticated_weights : Weights}
    {full_fit : Weights}
    {other_weights : Weights}
    {other_vector : Outputs}
    (is_a_mastication : IS_A_MASTICATION predict labels start_weights masticated_weights)
    (full_fit_fits : predict full_fit = labels)
    (other_weights_move_along_features :
      other_weights = start_weights + adjoint predict other_vector) :
    let masticated_distance := ‖masticated_weights - full_fit‖
    let other_distance  := ‖other_weights - full_fit‖
    masticated_distance ≤ other_distance := by
  unfold_named_pieces
  obtain ⟨mastication_fits, vector, weights_move_along_features⟩ := is_a_mastication
  have rest_is_unseen : predict (masticated_weights - full_fit) = 0 := by
    rw [map_sub, mastication_fits, full_fit_fits, sub_self]
  -- From the masticated weights, the other weights are a move along the features too.
  have other_from_masticated :
      other_weights = masticated_weights + adjoint predict (other_vector - vector) := by
    rw [other_weights_move_along_features, weights_move_along_features, map_sub]
    abel
  have right_angle :
      inner ℝ (other_weights - masticated_weights) (masticated_weights - full_fit) = 0 :=
    a_move_along_the_features_is_at_right_angles_to_what_the_batch_does_not_see
      predict other_from_masticated rest_is_unseen
  have split : other_weights - full_fit
      = (other_weights - masticated_weights) + (masticated_weights - full_fit) := by
    abel
  have squares : ‖other_weights - full_fit‖ ^ 2
      = ‖other_weights - masticated_weights‖ ^ 2 + ‖masticated_weights - full_fit‖ ^ 2 := by
    rw [split,
      norm_add_sq_real (other_weights - masticated_weights) (masticated_weights - full_fit),
      right_angle]
    ring
  by_contra other_is_nearer
  rw [not_le] at other_is_nearer
  nlinarith [norm_nonneg (other_weights - full_fit),
    norm_nonneg (masticated_weights - full_fit),
    sq_nonneg ‖other_weights - masticated_weights‖]

/-- **The squared travels add up to the distance gained.** Over any number of mastications, one
batch after another, the squared travels add up to the first squared distance to a full fit
minus the last. -/
theorem squared_travels_add_up_to_the_distance_gained
    (predict : ℕ → Weights →L[ℝ] Outputs)
    (labels : ℕ → Outputs)
    (weights : ℕ → Weights)
    {full_fit : Weights}
    (every_step_is_a_mastication : ∀ batch : ℕ,
      IS_A_MASTICATION (predict batch) (labels batch) (weights batch) (weights (batch + 1)))
    (full_fit_fits_every_batch : ∀ batch : ℕ, predict batch full_fit = labels batch)
    (mastication_count : ℕ) :
    let total_squared_travel :=
      ∑ batch ∈ Finset.range mastication_count, ‖weights (batch + 1) - weights batch‖ ^ 2
    let distance_at_start := ‖weights 0 - full_fit‖
    let distance_at_end   := ‖weights mastication_count - full_fit‖
    total_squared_travel = distance_at_start ^ 2 - distance_at_end ^ 2 := by
  unfold_named_pieces
  induction mastication_count with
  | zero => simp
  | succ previous_count earlier_case =>
    have this_mastication : ‖weights previous_count - full_fit‖ ^ 2
        = ‖weights (previous_count + 1) - full_fit‖ ^ 2
          + ‖weights (previous_count + 1) - weights previous_count‖ ^ 2 :=
      mastication_brings_weights_closer_by_exactly_its_travel (predict previous_count)
        (every_step_is_a_mastication previous_count) (full_fit_fits_every_batch previous_count)
    rw [Finset.sum_range_succ, earlier_case]
    linarith

/-- **The travel runs out.** However many mastications there are, their squared travels add up to
at most the first squared distance to a full fit. -/
theorem total_squared_travel_is_at_most_the_first_squared_distance
    (predict : ℕ → Weights →L[ℝ] Outputs)
    (labels : ℕ → Outputs)
    (weights : ℕ → Weights)
    {full_fit : Weights}
    (every_step_is_a_mastication : ∀ batch : ℕ,
      IS_A_MASTICATION (predict batch) (labels batch) (weights batch) (weights (batch + 1)))
    (full_fit_fits_every_batch : ∀ batch : ℕ, predict batch full_fit = labels batch)
    (mastication_count : ℕ) :
    let total_squared_travel :=
      ∑ batch ∈ Finset.range mastication_count, ‖weights (batch + 1) - weights batch‖ ^ 2
    let distance_at_start := ‖weights 0 - full_fit‖
    total_squared_travel ≤ distance_at_start ^ 2 := by
  unfold_named_pieces
  have gained : ∑ batch ∈ Finset.range mastication_count, ‖weights (batch + 1) - weights batch‖ ^ 2
      = ‖weights 0 - full_fit‖ ^ 2 - ‖weights mastication_count - full_fit‖ ^ 2 :=
    squared_travels_add_up_to_the_distance_gained predict labels weights
      every_step_is_a_mastication full_fit_fits_every_batch mastication_count
  rw [gained]
  linarith [sq_nonneg ‖weights mastication_count - full_fit‖]

omit [CompleteSpace OldOutputs] in
/-- **All the forgetting together is bounded.** The changes that the mastications make to the errors
on an old batch, squared and added up, are at most the squared size of the old predictor
times the first squared distance to a full fit. The ceiling does not grow with the number of
mastications. -/
theorem total_squared_forgetting_is_bounded
    (old_predict : Weights →L[ℝ] OldOutputs)
    (old_labels : OldOutputs)
    (predict : ℕ → Weights →L[ℝ] Outputs)
    (labels : ℕ → Outputs)
    (weights : ℕ → Weights)
    {full_fit : Weights}
    (every_step_is_a_mastication : ∀ batch : ℕ,
      IS_A_MASTICATION (predict batch) (labels batch) (weights batch) (weights (batch + 1)))
    (full_fit_fits_every_batch : ∀ batch : ℕ, predict batch full_fit = labels batch)
    (mastication_count : ℕ) :
    let linear_old_predict : LinearPredictor Weights OldOutputs := old_predict
    let total_squared_forgetting :=
      ∑ batch ∈ Finset.range mastication_count,
        ‖GET_RESIDUAL linear_old_predict old_labels (weights (batch + 1))
          - GET_RESIDUAL linear_old_predict old_labels (weights batch)‖ ^ 2
    let distance_at_start := ‖weights 0 - full_fit‖
    let ceiling           := ‖old_predict‖ ^ 2 * distance_at_start ^ 2
    total_squared_forgetting ≤ ceiling := by
  unfold_named_pieces
  have square_of_a_bound : ∀ change size travel : ℝ,
      0 ≤ change → change ≤ size * travel → change ^ 2 ≤ size ^ 2 * travel ^ 2 := by
    intro change size travel change_nonneg change_le
    nlinarith [sq_nonneg (size * travel - change),
      mul_nonneg change_nonneg (sub_nonneg.mpr change_le)]
  have travel_runs_out :
      ∑ batch ∈ Finset.range mastication_count, ‖weights (batch + 1) - weights batch‖ ^ 2
        ≤ ‖weights 0 - full_fit‖ ^ 2 :=
    total_squared_travel_is_at_most_the_first_squared_distance predict labels weights
      every_step_is_a_mastication full_fit_fits_every_batch mastication_count
  calc ∑ batch ∈ Finset.range mastication_count,
        ‖GET_RESIDUAL (old_predict : Weights →ₗ[ℝ] OldOutputs) old_labels (weights (batch + 1))
          - GET_RESIDUAL (old_predict : Weights →ₗ[ℝ] OldOutputs) old_labels (weights batch)‖ ^ 2
      ≤ ∑ batch ∈ Finset.range mastication_count,
          ‖old_predict‖ ^ 2 * ‖weights (batch + 1) - weights batch‖ ^ 2 := by
        apply Finset.sum_le_sum
        intro batch _
        exact square_of_a_bound _ _ _ (norm_nonneg _)
          (forgetting_is_at_most_predict_size_times_distance old_predict old_labels
            (weights batch) (weights (batch + 1)))
    _ = ‖old_predict‖ ^ 2
          * ∑ batch ∈ Finset.range mastication_count, ‖weights (batch + 1) - weights batch‖ ^ 2 := by
        rw [Finset.mul_sum]
    _ ≤ ‖old_predict‖ ^ 2 * ‖weights 0 - full_fit‖ ^ 2 :=
        mul_le_mul_of_nonneg_left travel_runs_out (sq_nonneg _)

end Mastication

/-! ### A mastication as a probe: what one batch teaches about another -/

section Probe

open ContinuousLinearMap

variable {Weights Outputs OtherOutputs : Type*}
  [NormedAddCommGroup Weights] [InnerProductSpace ℝ Weights]
  [CompleteSpace Weights]
  [NormedAddCommGroup Outputs] [InnerProductSpace ℝ Outputs]
  [CompleteSpace Outputs]
  [NormedAddCommGroup OtherOutputs] [InnerProductSpace ℝ OtherOutputs]
  [CompleteSpace OtherOutputs]

/-- **A batch that already fits needs no travel.** If the start weights fit the batch, a mastication
of that batch leaves them where they are. A duplicate of a batch already masticated is such a
batch. -/
theorem masticating_a_batch_that_already_fits_moves_nothing
    (predict : Weights →L[ℝ] Outputs)
    {labels : Outputs}
    {start_weights : Weights}
    {masticated_weights : Weights}
    (start_already_fits : predict start_weights = labels)
    (is_a_mastication : IS_A_MASTICATION predict labels start_weights masticated_weights) :
    masticated_weights = start_weights := by
  obtain ⟨mastication_fits, vector, weights_move_along_features⟩ := is_a_mastication
  -- Both ends fit the batch, so the move is unseen by the batch.
  have move_is_unseen : predict (masticated_weights - start_weights) = 0 := by
    rw [map_sub, mastication_fits, start_already_fits, sub_self]
  -- It is also along the features, so it is at right angles to itself.
  have move_is_at_right_angles_to_itself :
      inner ℝ (masticated_weights - start_weights) (masticated_weights - start_weights) = 0 :=
    a_move_along_the_features_is_at_right_angles_to_what_the_batch_does_not_see
      predict weights_move_along_features move_is_unseen
  exact sub_eq_zero.mp (inner_self_eq_zero.mp move_is_at_right_angles_to_itself)

/-- **The travel of a mastication is at most the distance to a full fit.** -/
theorem travel_of_a_mastication_is_at_most_the_distance_to_a_full_fit
    (predict : Weights →L[ℝ] Outputs)
    {labels : Outputs}
    {start_weights : Weights}
    {masticated_weights : Weights}
    {full_fit : Weights}
    (is_a_mastication : IS_A_MASTICATION predict labels start_weights masticated_weights)
    (full_fit_fits : predict full_fit = labels) :
    let travel          := ‖masticated_weights - start_weights‖
    let distance_before := ‖start_weights - full_fit‖
    travel ≤ distance_before := by
  unfold_named_pieces
  have squares : ‖start_weights - full_fit‖ ^ 2
      = ‖masticated_weights - full_fit‖ ^ 2 + ‖masticated_weights - start_weights‖ ^ 2 :=
    mastication_brings_weights_closer_by_exactly_its_travel predict is_a_mastication full_fit_fits
  by_contra travel_is_longer
  rw [not_le] at travel_is_longer
  nlinarith [norm_nonneg (start_weights - full_fit),
    norm_nonneg (masticated_weights - start_weights),
    sq_nonneg ‖masticated_weights - full_fit‖]

omit [CompleteSpace Weights] [CompleteSpace Outputs] [CompleteSpace OtherOutputs] in
/-- **What the batch spans is learned with it.** Take another batch whose outputs are a
combination of the outputs of this batch, whatever the weights. If a full fit of both
exists, any weights that fit this batch fit the other batch too, with no step taken on it. -/
theorem what_the_batch_spans_is_learned_with_it
    (predict : Weights →L[ℝ] Outputs)
    (other_predict : Weights →L[ℝ] OtherOutputs)
    (combine : Outputs →L[ℝ] OtherOutputs)
    {labels : Outputs}
    {other_labels : OtherOutputs}
    {weights : Weights}
    {full_fit : Weights}
    (other_is_a_combination :
      ∀ any_weights : Weights, other_predict any_weights = combine (predict any_weights))
    (weights_fit : predict weights = labels)
    (full_fit_fits : predict full_fit = labels)
    (full_fit_fits_other : other_predict full_fit = other_labels) :
    other_predict weights = other_labels := by
  rw [other_is_a_combination, weights_fit, ← full_fit_fits, ← other_is_a_combination,
    full_fit_fits_other]

omit [CompleteSpace Outputs] in
/-- **A batch that is spanned needs no travel.** After the weights fit a batch, a mastication of
another batch that it spans moves nothing. -/
theorem a_batch_that_is_spanned_needs_no_travel
    (predict : Weights →L[ℝ] Outputs)
    (other_predict : Weights →L[ℝ] OtherOutputs)
    (combine : Outputs →L[ℝ] OtherOutputs)
    {labels : Outputs}
    {other_labels : OtherOutputs}
    {weights : Weights}
    {masticated_weights : Weights}
    {full_fit : Weights}
    (other_is_a_combination :
      ∀ any_weights : Weights, other_predict any_weights = combine (predict any_weights))
    (weights_fit : predict weights = labels)
    (full_fit_fits : predict full_fit = labels)
    (full_fit_fits_other : other_predict full_fit = other_labels)
    (is_a_mastication : IS_A_MASTICATION other_predict other_labels weights masticated_weights) :
    masticated_weights = weights :=
  masticating_a_batch_that_already_fits_moves_nothing other_predict
    (what_the_batch_spans_is_learned_with_it predict other_predict combine
      other_is_a_combination weights_fit full_fit_fits full_fit_fits_other)
    is_a_mastication

omit [CompleteSpace OtherOutputs] in
/-- **A batch with separate features is left alone.** If no move along the features of this
batch changes the outputs of another batch, a mastication of this batch leaves the outputs of the
other batch exactly where they were. -/
theorem masticating_a_batch_with_separate_features_changes_nothing_elsewhere
    (predict : Weights →L[ℝ] Outputs)
    (other_predict : Weights →L[ℝ] OtherOutputs)
    {start_weights : Weights}
    {moved_weights : Weights}
    {vector : Outputs}
    (weights_move_along_features :
      moved_weights = start_weights + adjoint predict vector)
    (features_are_separate :
      ∀ any_vector : Outputs, other_predict (adjoint predict any_vector) = 0) :
    other_predict moved_weights = other_predict start_weights := by
  rw [weights_move_along_features, map_add, features_are_separate, add_zero]

/-- **The error elsewhere is at most the distance times what sticks out.** Take one other
example with feature vector `feature`. Split it into a part that is a combination of the
features of the batch and the rest. For weights that fit the batch, the error on that
example is at most the size of the rest times the distance from the weights to a full fit. -/
theorem error_elsewhere_is_at_most_distance_times_what_sticks_out
    (predict : Weights →L[ℝ] Outputs)
    {labels : Outputs}
    {weights : Weights}
    {full_fit : Weights}
    {feature : Weights}
    {seen_part : Weights}
    {unseen_part : Weights}
    {coefficients : Outputs}
    (feature_splits : feature = seen_part + unseen_part)
    (seen_part_is_a_combination : seen_part = adjoint predict coefficients)
    (weights_fit : predict weights = labels)
    (full_fit_fits : predict full_fit = labels) :
    let error_elsewhere := |inner ℝ feature weights - inner ℝ feature full_fit|
    let sticking_out    := ‖unseen_part‖
    let distance        := ‖weights - full_fit‖
    error_elsewhere ≤ sticking_out * distance := by
  unfold_named_pieces
  have rest_is_unseen : predict (weights - full_fit) = 0 := by
    rw [map_sub, weights_fit, full_fit_fits, sub_self]
  -- The part of the feature that the batch spans sees no difference between the two.
  have only_what_sticks_out_counts :
      inner ℝ feature weights - inner ℝ feature full_fit
        = inner ℝ unseen_part (weights - full_fit) := by
    rw [← inner_sub_right, feature_splits, inner_add_left, seen_part_is_a_combination,
      adjoint_inner_left, rest_is_unseen, inner_zero_right, zero_add]
  rw [only_what_sticks_out_counts]
  exact abs_real_inner_le_norm _ _

end Probe

/-! ### One number fitted to a batch: a mastication keeps nothing of the past -/

section OneNumber

variable {Sample : Type*}

/-- The mean of the labels of a batch. -/
noncomputable def BATCH_MEAN (batch : Finset Sample) (labels : Sample → ℝ) : ℝ :=
  (∑ sample ∈ batch, labels sample) / batch.card

/-- **The loss of a guess is the loss of the batch mean plus the miss.** For one number
fitted to a batch, the squared error of any guess is that of the batch mean plus the number
of labels times the squared gap between the guess and the mean. -/
theorem loss_of_a_guess_is_loss_of_the_batch_mean_plus_the_miss
    (batch : Finset Sample)
    (labels : Sample → ℝ)
    (batch_is_not_empty : batch.Nonempty)
    (guess : ℝ) :
    let batch_mean   := BATCH_MEAN batch labels
    let loss_of_guess := ∑ sample ∈ batch, (guess - labels sample) ^ 2
    let loss_of_mean  := ∑ sample ∈ batch, (batch_mean - labels sample) ^ 2
    let miss          := (batch.card : ℝ) * (guess - batch_mean) ^ 2
    loss_of_guess = loss_of_mean + miss := by
  unfold_named_pieces
  have count_is_not_zero : (batch.card : ℝ) ≠ 0 := by
    exact_mod_cast batch_is_not_empty.card_pos.ne'
  have mean_times_count :
      BATCH_MEAN batch labels * batch.card = ∑ sample ∈ batch, labels sample := by
    unfold BATCH_MEAN
    field_simp
  have each : ∀ sample ∈ batch, (guess - labels sample) ^ 2
      = (BATCH_MEAN batch labels - labels sample) ^ 2
        + ((guess - BATCH_MEAN batch labels) ^ 2
            + 2 * (guess - BATCH_MEAN batch labels) * BATCH_MEAN batch labels)
        - 2 * (guess - BATCH_MEAN batch labels) * labels sample := by
    intro sample _
    ring
  rw [Finset.sum_congr rfl each, Finset.sum_sub_distrib, Finset.sum_add_distrib,
    Finset.sum_const, ← Finset.mul_sum, ← mean_times_count, nsmul_eq_mul]
  ring

/-- **The batch mean is the best single number.** No guess has a lower squared error on the
batch than the batch mean. So a mastication of one number to convergence ends at the batch mean,
whatever the start was. -/
theorem batch_mean_is_the_best_single_number
    (batch : Finset Sample)
    (labels : Sample → ℝ)
    (batch_is_not_empty : batch.Nonempty)
    (guess : ℝ) :
    let batch_mean    := BATCH_MEAN batch labels
    let loss_of_guess := ∑ sample ∈ batch, (guess - labels sample) ^ 2
    let loss_of_mean  := ∑ sample ∈ batch, (batch_mean - labels sample) ^ 2
    loss_of_mean ≤ loss_of_guess := by
  unfold_named_pieces
  have split : ∑ sample ∈ batch, (guess - labels sample) ^ 2
      = ∑ sample ∈ batch, (BATCH_MEAN batch labels - labels sample) ^ 2
        + (batch.card : ℝ) * (guess - BATCH_MEAN batch labels) ^ 2 :=
    loss_of_a_guess_is_loss_of_the_batch_mean_plus_the_miss batch labels
      batch_is_not_empty guess
  rw [split]
  have miss_is_not_negative :
      0 ≤ (batch.card : ℝ) * (guess - BATCH_MEAN batch labels) ^ 2 :=
    mul_nonneg (Nat.cast_nonneg _) (sq_nonneg _)
  linarith

end OneNumber

end ThoroughMastication
