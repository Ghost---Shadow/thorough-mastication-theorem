# Chapters that are supersets: plan, predictions and stop rule, written before any run

Written on 26 Nov 2026. No code for this test existed when this file was committed.

## The question

The author's point, from Sal Khan: in school, errors compound. If you do not understand
chapter 3, you cannot understand chapter 4. And school chapters are supersets: chapter 1 has
the concepts of chapter 1, chapter 2 has the concepts of chapters 1 and 2, and chapter N has
the concepts of chapters 1 to N.

This test asks what that structure does in the setting the theorems cover, a model that is
linear in its weights. It does **not** test compounding in a model that learns its own
features; that needs a chain model and is not built yet.

## What is run

- **Concepts:** 96 weights in four blocks of 24. Block `k` is the new material of chapter `k`.
- **Superset chapters:** an example of chapter `k` has random inputs in the first `24 k`
  coordinates and zeros after. Chapter `k` has `32 k` examples, enough to span all of its
  concepts.
- **Disjoint chapters, for contrast:** an example of chapter `k` has random inputs in block
  `k` only. Chapter `k` has 32 examples.
- **Labels:** from a linear teacher, with no noise.
- **Model:** linear, 96 weights, starting at zero, squared error. Float64, on the processor.
- **A mastication:** gradient descent on the whole chapter until its loss is at most one
  millionth of what it was, or 200,000 steps, at a learning rate of one over the largest
  strength of the chapter's kernel.
- **The table:** masticate each chapter from the start and measure every other chapter, as in
  cross-mastication. `entry(k, j)` is the share of chapter `j`'s loss removed by chapter `k`.
- **The mastery pass:** masticate chapters 1, 2, 3, 4 in order without going back to the start.
- **The time-boxed pass:** give each chapter 5 gradient steps, in order, and move on.
- **Seeds:** 0 to 4. A prediction holds when it is true in at least 4 of the 5 seeds.

## Predictions for superset chapters

- **S1, a later chapter teaches all the earlier ones.** `entry(k, j)` is at least 0.99
  whenever `j` is before `k`.
- **S2, an earlier chapter teaches only part of a later one.** `entry(k, j)` is above 0 and at
  most 0.9 whenever `j` is after `k`. (The theory says about `k / j`.)
- **S3, the table gives the order.** The rule "`i` comes before `j` when `entry(j, i)` is at
  least 0.99 and `entry(i, j)` is at most 0.9" gives exactly the order 1, 2, 3, 4: all six
  pairs, and no pair the wrong way.
- **S4, mastery in order forgets nothing.** After the mastery pass, the loss on every chapter
  is at most one ten-thousandth of its loss at the start.
- **S5, each chapter costs only its new material.** In the mastery pass, the squared travels
  of the four mastications add up to the squared distance from the start to the teacher,
  within 1%.
- **S6, time-boxing is forgiven.** In the time-boxed pass, the loss on chapter 1 at the end of
  the pass is lower than it was right after chapter 1's own 5 steps. Later chapters contain
  its concepts, so they keep teaching it.

## Prediction for disjoint chapters

- **S7, nothing is revisited.** In the time-boxed pass, the loss on chapter 1 at the end of
  the pass equals its loss right after its own 5 steps, to one part in a billion. What was
  not learned in its time stays unlearned.

## What the predictions say about the question, if they hold

In this setting supersets do the opposite of compounding: every later chapter is a second
chance at the earlier ones (S6), and the last chapter alone teaches everything (S1). Disjoint
chapters give no second chance (S7), but they do not depend on each other either. So in a
model with fixed features, neither layout makes chapter 4 impossible without chapter 3. If
Khan's compounding is real for a learner, it has to come from what the learner builds on the
way, which is the chain model.

## Stop rule

One run: both layouts, five seeds. Nothing is changed and no seed is added after the results
are seen. A test of one seed may be run first to see that the code does not crash; its
numbers are thrown away.
