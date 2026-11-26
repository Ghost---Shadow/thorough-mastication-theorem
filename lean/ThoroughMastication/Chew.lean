/-
Chewing a batch: training on one batch until it fits, then the next.

The plain English of every proof is in `docs/proofs.md`.

For a model linear in its weights:
* `gradient_descent_to_a_fit_is_a_chew` : gradient descent that arrives at a fit of a batch
  is a chew.
* `chew_brings_weights_closer_by_exactly_its_travel` : the squared distance to a full fit
  falls by the squared travel of the chew.
* `no_use_of_the_batch_gets_nearer_than_the_chew` : no other move along the features of the
  batch ends nearer to a full fit.
* `squared_travels_add_up_to_the_distance_gained` : over many chews, the squared travels add
  up to the squared distance gained.
* `total_squared_forgetting_is_bounded` : the changes in the errors of an old batch, squared
  and added up over all chews, stay under a ceiling that does not grow with the chews.

For one number fitted to a batch:
* `batch_mean_is_the_best_single_number` : the batch mean is the best guess, so a chew to
  convergence keeps nothing of the start.
-/
import SquigglyLine.Supporting.Training.LeastAction
import SquigglyLine.Supporting.NewInputs.Forgetting

namespace ThoroughMastication

open SquigglyLine

/-! ### A chew, and what it does to the distance to a full fit -/

section Chew

open ContinuousLinearMap

variable {Weights Outputs OldOutputs : Type*}
  [NormedAddCommGroup Weights] [InnerProductSpace ℝ Weights]
  [CompleteSpace Weights]
  [NormedAddCommGroup Outputs] [InnerProductSpace ℝ Outputs]
  [CompleteSpace Outputs]
  [NormedAddCommGroup OldOutputs] [InnerProductSpace ℝ OldOutputs]
  [CompleteSpace OldOutputs]

/-- A chew of one batch: `chewed_weights` fit the batch and are reached from
`start_weights` by a move along the features of the batch. -/
def IS_A_CHEW
    (predict : Weights →L[ℝ] Outputs)
    (labels : Outputs)
    (start_weights : Weights)
    (chewed_weights : Weights) : Prop :=
  predict chewed_weights = labels ∧
    ∃ vector : Outputs, chewed_weights = start_weights + adjoint predict vector

/-- **Gradient descent to a fit is a chew.** If gradient descent on a batch arrives at a fit
of the batch, the weights it arrives at are a chew of the start weights. -/
theorem gradient_descent_to_a_fit_is_a_chew
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
    IS_A_CHEW predict labels start_weights trained_weights := by
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

/-- **A chew brings the weights closer to a full fit by exactly its travel.** The squared
distance to any weights that fit the batch falls by the squared travel of the chew. -/
theorem chew_brings_weights_closer_by_exactly_its_travel
    (predict : Weights →L[ℝ] Outputs)
    {labels : Outputs}
    {start_weights : Weights}
    {chewed_weights : Weights}
    {full_fit : Weights}
    (is_a_chew : IS_A_CHEW predict labels start_weights chewed_weights)
    (full_fit_fits : predict full_fit = labels) :
    let distance_before := ‖start_weights - full_fit‖
    let distance_after  := ‖chewed_weights - full_fit‖
    let travel          := ‖chewed_weights - start_weights‖
    distance_before ^ 2 = distance_after ^ 2 + travel ^ 2 := by
  unfold_named_pieces
  obtain ⟨chew_fits, vector, weights_move_along_features⟩ := is_a_chew
  -- Both fit the batch, so the way between them is unseen by the batch.
  have rest_is_unseen : predict (chewed_weights - full_fit) = 0 := by
    rw [map_sub, chew_fits, full_fit_fits, sub_self]
  have right_angle :
      inner ℝ (chewed_weights - full_fit) (chewed_weights - start_weights) = 0 := by
    rw [real_inner_comm]
    exact a_move_along_the_features_is_at_right_angles_to_what_the_batch_does_not_see
      predict weights_move_along_features rest_is_unseen
  have split : start_weights - full_fit
      = (chewed_weights - full_fit) - (chewed_weights - start_weights) := by
    abel
  -- Pythagoras.
  rw [split,
    norm_sub_sq_real (chewed_weights - full_fit) (chewed_weights - start_weights),
    right_angle]
  ring

/-- **A chew never moves the weights away from a full fit.** -/
theorem chew_never_moves_weights_away_from_a_full_fit
    (predict : Weights →L[ℝ] Outputs)
    {labels : Outputs}
    {start_weights : Weights}
    {chewed_weights : Weights}
    {full_fit : Weights}
    (is_a_chew : IS_A_CHEW predict labels start_weights chewed_weights)
    (full_fit_fits : predict full_fit = labels) :
    let distance_before := ‖start_weights - full_fit‖
    let distance_after  := ‖chewed_weights - full_fit‖
    distance_after ≤ distance_before := by
  unfold_named_pieces
  have squares : ‖start_weights - full_fit‖ ^ 2
      = ‖chewed_weights - full_fit‖ ^ 2 + ‖chewed_weights - start_weights‖ ^ 2 :=
    chew_brings_weights_closer_by_exactly_its_travel predict is_a_chew full_fit_fits
  by_contra moved_away
  rw [not_le] at moved_away
  nlinarith [norm_nonneg (start_weights - full_fit),
    norm_nonneg (chewed_weights - full_fit),
    sq_nonneg ‖chewed_weights - start_weights‖]

/-- **No use of the batch gets nearer than the chew.** Any other weights reached from the
same start by a move along the features of the batch, fitting it or not, are no nearer to a
full fit than the chewed weights. A single gradient step and a chew stopped early are such
moves. -/
theorem no_use_of_the_batch_gets_nearer_than_the_chew
    (predict : Weights →L[ℝ] Outputs)
    {labels : Outputs}
    {start_weights : Weights}
    {chewed_weights : Weights}
    {full_fit : Weights}
    {other_weights : Weights}
    {other_vector : Outputs}
    (is_a_chew : IS_A_CHEW predict labels start_weights chewed_weights)
    (full_fit_fits : predict full_fit = labels)
    (other_weights_move_along_features :
      other_weights = start_weights + adjoint predict other_vector) :
    let chewed_distance := ‖chewed_weights - full_fit‖
    let other_distance  := ‖other_weights - full_fit‖
    chewed_distance ≤ other_distance := by
  unfold_named_pieces
  obtain ⟨chew_fits, vector, weights_move_along_features⟩ := is_a_chew
  have rest_is_unseen : predict (chewed_weights - full_fit) = 0 := by
    rw [map_sub, chew_fits, full_fit_fits, sub_self]
  -- From the chewed weights, the other weights are a move along the features too.
  have other_from_chewed :
      other_weights = chewed_weights + adjoint predict (other_vector - vector) := by
    rw [other_weights_move_along_features, weights_move_along_features, map_sub]
    abel
  have right_angle :
      inner ℝ (other_weights - chewed_weights) (chewed_weights - full_fit) = 0 :=
    a_move_along_the_features_is_at_right_angles_to_what_the_batch_does_not_see
      predict other_from_chewed rest_is_unseen
  have split : other_weights - full_fit
      = (other_weights - chewed_weights) + (chewed_weights - full_fit) := by
    abel
  have squares : ‖other_weights - full_fit‖ ^ 2
      = ‖other_weights - chewed_weights‖ ^ 2 + ‖chewed_weights - full_fit‖ ^ 2 := by
    rw [split,
      norm_add_sq_real (other_weights - chewed_weights) (chewed_weights - full_fit),
      right_angle]
    ring
  by_contra other_is_nearer
  rw [not_le] at other_is_nearer
  nlinarith [norm_nonneg (other_weights - full_fit),
    norm_nonneg (chewed_weights - full_fit),
    sq_nonneg ‖other_weights - chewed_weights‖]

/-- **The squared travels add up to the distance gained.** Over any number of chews, one
batch after another, the squared travels add up to the first squared distance to a full fit
minus the last. -/
theorem squared_travels_add_up_to_the_distance_gained
    (predict : ℕ → Weights →L[ℝ] Outputs)
    (labels : ℕ → Outputs)
    (weights : ℕ → Weights)
    {full_fit : Weights}
    (every_step_is_a_chew : ∀ batch : ℕ,
      IS_A_CHEW (predict batch) (labels batch) (weights batch) (weights (batch + 1)))
    (full_fit_fits_every_batch : ∀ batch : ℕ, predict batch full_fit = labels batch)
    (chew_count : ℕ) :
    let total_squared_travel :=
      ∑ batch ∈ Finset.range chew_count, ‖weights (batch + 1) - weights batch‖ ^ 2
    let distance_at_start := ‖weights 0 - full_fit‖
    let distance_at_end   := ‖weights chew_count - full_fit‖
    total_squared_travel = distance_at_start ^ 2 - distance_at_end ^ 2 := by
  unfold_named_pieces
  induction chew_count with
  | zero => simp
  | succ previous_count earlier_case =>
    have this_chew : ‖weights previous_count - full_fit‖ ^ 2
        = ‖weights (previous_count + 1) - full_fit‖ ^ 2
          + ‖weights (previous_count + 1) - weights previous_count‖ ^ 2 :=
      chew_brings_weights_closer_by_exactly_its_travel (predict previous_count)
        (every_step_is_a_chew previous_count) (full_fit_fits_every_batch previous_count)
    rw [Finset.sum_range_succ, earlier_case]
    linarith

/-- **The travel runs out.** However many chews there are, their squared travels add up to
at most the first squared distance to a full fit. -/
theorem total_squared_travel_is_at_most_the_first_squared_distance
    (predict : ℕ → Weights →L[ℝ] Outputs)
    (labels : ℕ → Outputs)
    (weights : ℕ → Weights)
    {full_fit : Weights}
    (every_step_is_a_chew : ∀ batch : ℕ,
      IS_A_CHEW (predict batch) (labels batch) (weights batch) (weights (batch + 1)))
    (full_fit_fits_every_batch : ∀ batch : ℕ, predict batch full_fit = labels batch)
    (chew_count : ℕ) :
    let total_squared_travel :=
      ∑ batch ∈ Finset.range chew_count, ‖weights (batch + 1) - weights batch‖ ^ 2
    let distance_at_start := ‖weights 0 - full_fit‖
    total_squared_travel ≤ distance_at_start ^ 2 := by
  unfold_named_pieces
  have gained : ∑ batch ∈ Finset.range chew_count, ‖weights (batch + 1) - weights batch‖ ^ 2
      = ‖weights 0 - full_fit‖ ^ 2 - ‖weights chew_count - full_fit‖ ^ 2 :=
    squared_travels_add_up_to_the_distance_gained predict labels weights
      every_step_is_a_chew full_fit_fits_every_batch chew_count
  rw [gained]
  linarith [sq_nonneg ‖weights chew_count - full_fit‖]

omit [CompleteSpace OldOutputs] in
/-- **All the forgetting together is bounded.** The changes that the chews make to the errors
on an old batch, squared and added up, are at most the squared size of the old predictor
times the first squared distance to a full fit. The ceiling does not grow with the number of
chews. -/
theorem total_squared_forgetting_is_bounded
    (old_predict : Weights →L[ℝ] OldOutputs)
    (old_labels : OldOutputs)
    (predict : ℕ → Weights →L[ℝ] Outputs)
    (labels : ℕ → Outputs)
    (weights : ℕ → Weights)
    {full_fit : Weights}
    (every_step_is_a_chew : ∀ batch : ℕ,
      IS_A_CHEW (predict batch) (labels batch) (weights batch) (weights (batch + 1)))
    (full_fit_fits_every_batch : ∀ batch : ℕ, predict batch full_fit = labels batch)
    (chew_count : ℕ) :
    let linear_old_predict : LinearPredictor Weights OldOutputs := old_predict
    let total_squared_forgetting :=
      ∑ batch ∈ Finset.range chew_count,
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
      ∑ batch ∈ Finset.range chew_count, ‖weights (batch + 1) - weights batch‖ ^ 2
        ≤ ‖weights 0 - full_fit‖ ^ 2 :=
    total_squared_travel_is_at_most_the_first_squared_distance predict labels weights
      every_step_is_a_chew full_fit_fits_every_batch chew_count
  calc ∑ batch ∈ Finset.range chew_count,
        ‖GET_RESIDUAL (old_predict : Weights →ₗ[ℝ] OldOutputs) old_labels (weights (batch + 1))
          - GET_RESIDUAL (old_predict : Weights →ₗ[ℝ] OldOutputs) old_labels (weights batch)‖ ^ 2
      ≤ ∑ batch ∈ Finset.range chew_count,
          ‖old_predict‖ ^ 2 * ‖weights (batch + 1) - weights batch‖ ^ 2 := by
        apply Finset.sum_le_sum
        intro batch _
        exact square_of_a_bound _ _ _ (norm_nonneg _)
          (forgetting_is_at_most_predict_size_times_distance old_predict old_labels
            (weights batch) (weights (batch + 1)))
    _ = ‖old_predict‖ ^ 2
          * ∑ batch ∈ Finset.range chew_count, ‖weights (batch + 1) - weights batch‖ ^ 2 := by
        rw [Finset.mul_sum]
    _ ≤ ‖old_predict‖ ^ 2 * ‖weights 0 - full_fit‖ ^ 2 :=
        mul_le_mul_of_nonneg_left travel_runs_out (sq_nonneg _)

end Chew

/-! ### One number fitted to a batch: a chew keeps nothing of the past -/

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
batch than the batch mean. So a chew of one number to convergence ends at the batch mean,
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
