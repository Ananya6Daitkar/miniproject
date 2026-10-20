# Five-minute demonstration and viva

1. Start the server using README instructions. Open the Simulation tab. Explain the synthetic 50 km corridor and the one-machine assumption.
2. Click **Run comparison**. Select Trained SAC, press Play, and explain the changing track condition and selected tamping action.
3. Inspect the action probabilities and time feasibility. A probability is a policy output, not a calibrated confidence score.
4. Switch to CP-SAT and Threshold rule. Compare all metrics, not only cost. If a baseline wins, say so and discuss the difficulty of learning with limited samples.
5. Choose stress deterioration and a different seed, rerun, then export the CSV.
6. Open **Training & results**. Show actual learning curves and held-out results. Point out the separation of training, validation and testing.
7. Open **How it works**, then show `docs/IMPLEMENTATION.md` and the paper link.

## Short explanation

“Our project uses discrete Soft Actor-Critic, a deep reinforcement-learning algorithm, to learn when and where to maintain railway tracks. A simulator models deterioration, travel and work windows. We compare the learned policy with CP-SAT and a maintenance rule on identical unseen scenarios. Our data and parameters are synthetic, so this is a documented, scaled-down implementation rather than an exact reproduction.”

## Common viva questions

**Is this ML?** Yes. Neural-network parameters change through learning from rewards and sampled experience. SAC is deep reinforcement learning.

**Why not classification?** Decisions change future track condition and machine position; they are sequential rather than independent labels.

**Why two critics?** Taking the lower estimate reduces overestimation of action values.

**Why replay?** It reuses earlier experiences for off-policy learning and reduces dependence on consecutive samples.

**What does entropy do?** It discourages premature collapse to one action during learning. The implementation uses a fixed coefficient.

**What did you change from the paper?** Synthetic ten-unit environment, independently chosen numerical parameters, a documented network size, and reduced comparison scope. Full equations and original data were unavailable.

**What if a rule wins?** That is a valid result. It may indicate insufficient training, a strong problem-specific rule, or that the simplified task does not justify a complex learned policy.

**Why can all policies have zero invalid actions but different safety-related outcomes?** Feasibility masks enforce action constraints; overdue track exposure is a different outcome metric. Real railway safety is outside the simulation.
