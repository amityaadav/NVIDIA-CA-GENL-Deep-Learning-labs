# Product steering — Watch a Neural Network Think

## What it is
An interactive, educational visualizer of a forward pass through the Lab 1 MNIST
network. The user draws a digit; the app normalizes it the way MNIST expects,
runs one inference, and **replays the forward pass as an animation** — neurons
light up by activation, layer by layer, and the strongest weighted paths
(`weight × activation` on this specific drawing) trace how the answer forms.

## Why it exists
It's the "show, don't tell" companion to the labs. The labs teach the concepts;
this makes a single forward pass legible — you can *see* activations propagate
and see which connections actually drove the prediction. The animation, not the
compute, is the point: inference is sub-millisecond, deliberately paced out over
several seconds so a human can follow it.

## Who it's for
- Learners (including the author) building intuition about neural networks.
- Someone browsing a portfolio/demo — it must feel instant and self-explanatory
  with zero setup beyond "draw a digit, press Run."

## Core user flow
1. **Draw** a digit on the canvas (mouse/trackpad/touch).
2. **Run inference** (button, or Enter). Clear with the button or Escape.
3. **Watch** the forward pass animate; scrub with play/pause/step/speed controls.
4. Read the **prediction** and per-digit confidence.

## Product principles (use these to resolve design questions)
- **Legibility over completeness.** Show the strongest ~40 edges per transition,
  not all 400k. Every visual element should mean something (brightness =
  activation, color = excitatory/inhibitory, width/opacity = contribution
  strength).
- **Faithful to the real computation.** Highlighted paths reflect the actual
  contribution on *this* input, not decoration. Never fake the visualization.
- **Instant and frictionless.** No login, no upload, no config. First paint should
  invite drawing immediately.
- **Accurate predictions depend on preprocessing.** The MNIST-style normalization
  in the browser is load-bearing product behavior, not an implementation detail —
  don't weaken it to simplify code.

## In scope (natural next refinements)
- Better/altered visualizations of the same forward pass.
- Playback/interaction polish, accessibility, responsive layout.
- Explanatory UI (tooltips, legends, annotations).
- Swapping in other small nets that fit the same trace contract.

## Out of scope (unless the goal explicitly changes)
- Training in the browser or a training UI (training is an offline one-shot script).
- Datasets other than MNIST-shaped 28×28 grayscale digits.
- Accounts, persistence, multi-user features, analytics.
- Large/deep models where a full activation trace can't be drawn legibly.

## Definition of done for a change
Predictions stay accurate, the animation stays faithful to the computation, the
shared contract stays consistent across backend and frontend, and the thing still
works with no setup beyond the documented run commands.
