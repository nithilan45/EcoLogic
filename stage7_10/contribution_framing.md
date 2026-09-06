# Contribution-type framing

Scaffolding for an eventual paper's introduction, in the style NeurIPS asks
submissions to declare. One paragraph, not a paper.

---

This submission's primary contribution is an **evaluation-methodology
contribution carried by a negative result** — specifically: we audit a deployed
three-tier LLM query router against static, oracle and knapsack-optimal
baselines on 364 objectively auto-graded items, find that it is dominated on
both axes simultaneously by the trivial policy of sending every query to the
middle tier (86.8% vs 92.3% accuracy at 689 J vs 394 J, McNemar p = 0.0095
against the frontier and p = 0.25 against always-cheapest — i.e. its keyword
logic adds nothing over ignoring the query), show that this is not fixed by
replacing the keyword rules with a pre-registered learned router even after
scaling training data ~4.2× and de-noising its labels with k=3 majority voting,
and prove along the way that the confusion-matrix cost accounting the routing
literature uses to report savings is biased by an exactly characterisable
covariance term — `R_true = R_naive + Σ_{i≠j} Cov(1{t*=i, t̂=j}, e_j(x) − e_i(x))`
— which on our data is 1.9× the size of the regret being reported, flips its
sign, and reproduces as a sign flip on RouteLLM's own released GSM8K data, where
naive accounting overstates savings by up to 6.3%. The methodological
contribution is the audit protocol that makes such a finding falsifiable rather
than rhetorical: static single-tier baselines as the bar a router must clear,
pre-registered success criteria with one-shot frozen test sets, MCKP frontiers
separating *calibration* gaps (the router's fault) from *discreteness* gaps (the
problem's), and a variance decomposition showing that temperature-0 evaluation
is not deterministic and that part of every reported interval is regeneration
noise no amount of additional items would reduce. We claim no new model,
architecture or algorithm, and we do not claim that learned routing cannot
work — we tested one family of zero-API-cost routers on one workload; we claim
that energy-saving routers are currently being reported without the baselines or
the cost accounting needed to know whether they save anything.

---

## Notes on this framing (not part of the paragraph)

Three things it deliberately does:

- **Leads with the audit, not the derivation.** The Stage 8 correction is the
  most portable result, but it is a corollary of the definition of covariance.
  Presenting it as the headline would overclaim its depth. Presenting it as
  something the audit *forced us to notice* is both honest and stronger: the
  motivation is that two correct-looking computations of the same quantity
  disagreed in sign.
- **States the negative result as a measurement, not a failure.** "The router
  is beaten by a constant" is a finding about the router. The learned-router
  stages exist to establish that the finding is about the workload's
  predictability rather than about keyword matching being a weak
  implementation.
- **Bounds its own generality explicitly.** The last clause is the actual
  claim; everything before it is evidence. Reviewers punish
  negative-result papers that quietly generalise, and the honest scope here —
  one system, one substituted model stack, four academic benchmarks — cannot
  support "routing does not work".

The weakest point a reviewer will find, and it should be conceded in the
introduction rather than defended: **no joule was measured**. Energy is a linear
transform of measured token counts under the audited system's own assumed
rates. The ±5× sensitivity sweep bounds the conclusion's robustness to those
rates, but the paper cannot claim a physical energy measurement, and the
Stage 8/9 contribution is best framed as being about *cost accounting under
per-item cost dispersion* — which is rate-model independent — rather than about
energy specifically.
