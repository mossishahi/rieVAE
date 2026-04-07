# Unified Action Plan: SCR-VAE Paper Revision
## Synthesized from all four committee reviews (Gem31, op46, Op46MX, Snn46)

---

## Consensus Across All 4 Reviewers

These findings were raised by every reviewer and must be fixed before submission:

| # | Finding | Source | Priority |
|---|---|---|---|
| C1 | Torus experiment uses WRONG manifold (round $\mathbb{R}^3$ torus ≠ flat torus) | All 4 | **CRITICAL** |
| C2 | Curvature proxy theoretical prediction inconsistent (Eq.(16) vs Remark 8) | All 4 | **CRITICAL** |
| C3 | A4' (metric approximation) not derived from first principles | All 4 | **High** |
| C4 | Edge KL term absent from theoretical loss Equation (3) | 3/4 | **Moderate** |

---

## Action Items by Category

### TIER 1: Critical — Must fix before submission

#### T1.1 — Fix the torus experiment (Clifford embedding)

**Problem**: `flat_torus()` in `data/synthetic.py` generates the standard
embedded torus in $\mathbb{R}^3$ with Gaussian curvature
$K(\phi) = \cos\phi / [r(R+r\cos\phi)]$ ranging from $+1/3$ to $-1$ for
$R=2, r=1$. The geodesic evaluation uses the flat formula
$d = \sqrt{(R\Delta\theta)^2 + (r\Delta\phi)^2}$ — a metric mismatch
causing up to 125% relative error at the inner equator.

**Fix**: Switch to the Clifford torus in $\mathbb{R}^4$:
$$f(\theta,\phi) = (R\cos\theta, R\sin\theta, r\cos\phi, r\sin\phi)$$
This embedding is genuinely flat ($K=0$ everywhere), and the flat geodesic
formula is EXACT.

**Code change**: Add `flat_torus_clifford()` to `data/synthetic.py`, update
embedding matrix to $A \in \mathbb{R}^{G \times 4}$, rerun all torus experiments.

**Expected result**: After this fix, the curvature proxy $\hat{\kappa}$
should approach 0 (confirming the curvature detection works as a negative test),
and isometry evaluation will use the correct ground truth.

#### T1.2 — Resolve the curvature proxy formula

**Problem**: Equation (16) claims $\hat{\kappa} \to 2/R$ (scale-invariant).
Remark 8 correctly derives $\hat{\kappa} \to (2/R)\sqrt{3/G}$ for random
embeddings. For $R=1, G=50$: $2/R = 2.000$ vs. $(2/R)\sqrt{3/G} = 0.4899$.
These are inconsistent by a factor of $\sqrt{G/3} \approx 4$.

**Resolution (from Snn46 — the KEY THEORETICAL INSIGHT)**: Both formulas
are correct in different regimes:
- For an **uncalibrated decoder** (scale factor $\alpha \neq \sqrt{3/G}$):
  $\hat{\kappa} \to (2/R\alpha)\sqrt{3/G}$
- At the **isometric fixed point** (where $\alpha = \sqrt{3/G}$, i.e.,
  pullback metric = true metric): $\hat{\kappa} \to 2/R = 2\sqrt{K}$

**Consequence**: $\hat{\kappa}$ is simultaneously:
1. A **curvature detector**: $\hat{\kappa} = 0 \Leftrightarrow K = 0$ locally
2. An **isometry quality metric**: $\hat{\kappa} \to 2\sqrt{K}$ if and only if
   the decoder is isometrically calibrated

This is a major theoretical contribution that the paper missed! The experimental
target $\hat{\kappa}^* = 2.0$ is the isometrically-calibrated value (correct).
The baseline's $\hat{\kappa} = 2.849 > 2.0$ overshoots because Euclidean KNN
introduces spurious curvature. SCR-VAE's $1.442 < 2.0$ shows the decoder is
moving toward isometric calibration but not yet there at finite $n$ and $T$.

**Action**:
1. Remove Equation (16)'s claim of scale-invariance (it is only true at isometric FP)
2. Add Proposition (Snn46's Prop. 4): "κ̂ → 2√K iff decoder is isometrically calibrated"
3. Rephrase Table 1: the "True" column for sphere is 2.0 (isometric target), not the
   uncalibrated prediction 0.49

#### T1.3 — Fix Lemma 1 proof (J_φ ≠ I_d)

**Problem**: The proof claims "$J_\varphi(z_i) = I_d$ at the base point of
normal coordinates." This is FALSE. The Jacobian of the coordinate chart is
$J_\varphi(z_i) = G(z_i)^{-1/2}Q$ for orthogonal $Q$, which equals $I_d$
only when $G(z_i) = I_d$. The RESULT is still correct but the proof step is
wrong.

**Corrected proof** (from op46): Use the geodesic ODE. The geodesic velocity
$v = \exp_{z_i}^{-1}(z_j)$ satisfies:
$$z_j = z_i + v - \tfrac{1}{2}\Gamma^k_{lm}(z_i)v^l v^m \hat{e}_k + O(\|v\|^3)$$
Inverting: $v = \Delta z_{ij} + O(\|\Delta z\|^2)$.
The cross term $\langle \Delta z, c\rangle_{G(z_i)} = O(K_{\max}\|\Delta z\|^3)$
via the Christoffel-metric identity, giving $|d_R - w_{ij}| = O(K_{\max}r^3)$.

**Action**: Rewrite Step 3 of Lemma 1 using this argument.

---

### TIER 2: High Priority — Should fix before submission

#### T2.1 — Add Fixed-Point Existence Proof (Snn46 Gap 1)

**Problem**: Theorem 1 characterizes properties AT a self-consistent pair
$(f^*, G^*)$ but never proves such a pair exists. Proposition 3 (convergence)
starts "in a neighbourhood of $(f^*, G^*)$" — without guaranteeing it exists.

**Fix** (Snn46's bridge): The set of KNN graphs $\mathcal{G}$ on $n$ nodes is
finite. Under generic non-degeneracy (no tied Riemannian distances, which holds
with probability 1), the map $\Phi: G \mapsto \mathrm{KNN}(f^*(G))$ is a
self-map of a finite set, so it has a fixed point.

**Action**: Add Proposition (Fixed-Point Existence) before Theorem 1, using
the finite-set fixed-point argument.

#### T2.2 — Add Capacity Condition for A4' (all reviewers)

**Problem**: A4' is stated without derivation. The derivable bound is
$\|G^* - g\|_{\mathrm{op}} = O(\delta_{\mathrm{rec}}/r_n + r_n)$ which has
a $1/r_n$ factor that makes Theorem 2 vacuous for fixed $\delta_{\mathrm{rec}}$.

**Resolution**: Under $\delta_{\mathrm{rec}} = O(r_n^2)$ (high-capacity decoder),
$\delta_{\mathrm{rec}}/r_n = O(r_n)$ and A4' follows automatically.

**Action**: Add Lemma and Corollary after Assumption A4':
- Lemma: $\|J_{f^*} - J_{f_{\mathrm{true}}}\|_{\mathrm{op}} \leq 2\delta_{\mathrm{rec}}/r_n + C\kappa r_n$
- Corollary: If $\delta_{\mathrm{rec}} = O(r_n^2)$, then A4' holds and the isometry bound
  becomes $O(r_n + e^{-\lambda T/2})$ (three terms, no $\delta_{\mathrm{rec}}$ term).

#### T2.3 — Add Topological Obstruction Theorem (op46, Snn46)

**Problem**: The torus performance gap is explained informally but not
formalized. The paper needs a theorem that explains why the isometry floor
is a fundamental limitation, not an optimization failure.

**Fix** (from op46/Snn46): Any $C^1$ decoder $f: \mathbb{R}^2 \to \mathbb{R}^G$
approximating a compact manifold $M$ with $\pi_1(M) \neq 0$ must have either
incomplete coverage, metric degeneracy (folds), or curvature concentration.

**Action**: Add topological obstruction theorem (simplified version) to the
sphere-vs-torus discussion section. This turns the torus "failure" into a
positive result: the method correctly identifies the fundamental impossibility.

#### T2.4 — Fix Flat Torus Embeddability Claim (Snn46)

**Problem**: Section 5.5 states "flat torus cannot be isometrically embedded in
Euclidean space of any finite dimension smaller than dim M." This is FALSE.
The Clifford torus embeds isometrically in $\mathbb{R}^4$.

**Correct statement**: "cannot be smoothly isometrically embedded in $\mathbb{R}^3$
(Efimov's theorem)."

**Action**: One sentence fix in Section 5.5.

#### T2.5 — Reconcile Curvature Proxy Formulae (dual-regime explanation)

**Action** (follows from T1.2):
1. The curvature section currently states κ̂ → 2/R as "scale-invariant" (Eq. 16)
2. Remark 8 gives (2/R)√(3/G) for random embeddings
3. Add a Proposition showing both are correct for different regimes (isometric vs. uncalibrated)
4. Explain that Table 1's "True" column is the isometric target (correct use of 2/R)

---

### TIER 3: Moderate — Important but not blocking

#### T3.1 — Add Edge KL to Theoretical Loss (all reviewers)

**Problem**: Equation (3) has 3 terms; code uses 4 (plus edge KL $\beta_e \mathcal{L}_{\mathrm{edge\_KL}}$).

**Action**: Add the edge KL term to Equation (3) with note that $\beta_e = 10^{-3}$
makes it perturbative and does not affect the fixed-point theorem at leading order.

#### T3.2 — Fix Sphere Sampling (Op46MX)

**Problem**: Current sampling uses $\theta \sim \mathrm{Uniform}[0,\pi]$ which
gives non-uniform density on $S^2$. Uniform requires $\theta = \arccos(1-2U)$.

**Action**: One line fix in `data/synthetic.py`.

#### T3.3 — Fix Coefficient Error in Section 5.3 (Snn46)

**Problem**: The sphere verification writes $8R^2KA$ but it should be $4\sqrt{K}A$.

Verification: $4\sqrt{K} \cdot (\varepsilon^2/2) = 4/R \cdot \varepsilon^2/2 = 2\varepsilon^2/R$ ✓

**Action**: One line fix.

#### T3.4 — Add Empirical Verification of A4' (Op46MX)

**Action**: At the sphere experiment's fixed point, compute
$\|G^*(z_i) - g(z_i)\|_{\mathrm{op}}$ and plot vs. $r_n$.
If the scaling is $O(r_n)$ (not $O(\delta_{\mathrm{rec}}/r_n)$), this provides
empirical evidence that A4' holds in the overparameterized regime.

#### T3.5 — Document Implementation Details Not in Theory

**Actions**:
1. Add note in E-step: "Riemannian distances are symmetrized: $(w_{fwd}+w_{bwd})/2$,
   within $O(K_{\max}r^3)$ of either term alone (Lemma 1)."
2. Add note: "KNN candidates are restricted to current edges. Setting
   `n_extra_candidates > 0` allows discovery of new neighbors."
3. Change default `n_extra_candidates = 50` (currently 0).

---

### TIER 4: Minor Code Fixes

| ID | Fix | File |
|---|---|---|
| B1 | Fix broken test `edge_decoder.W.bias` → check named_parameters | `tests/test_model.py` |
| B2 | Fix code comment in `curvature.py` about κ̂ regime | `rieVAE/geometry/curvature.py` |
| B3 | Change default `n_extra_candidates = 50` | `rieVAE/train/trainer.py` |

---

### TIER 5: Future Work / Nice-to-Have (deferred)

| # | Item | Reviewer |
|---|---|---|
| F1 | Periodic latent space ($S^1 \times S^1$) for torus | Gem31, Snn46 |
| F2 | Compare to Arvanitidis et al. (2018) and geodesic VAE | op46 |
| F3 | Test on Swiss roll and higher-dimensional spheres | Op46MX |
| F4 | Non-contractible holonomy as topological certificate | Snn46 |
| F5 | Curvature-aware adaptive λ_riem | Op46MX |
| F6 | Multi-scale (coarse-to-fine) self-consistent iteration | Op46MX |
| F7 | Global convergence characterization | All |

---

## Summary: The Big Picture

Reading all four reviews together, the paper has **two fundamental issues**
and **one missed major insight**:

### Issue 1: The torus experiment is flawed (all agree)
The round torus ≠ flat torus. Switch to Clifford embedding. Estimated impact:
the curvature proxy on the torus should approach 0, and the isometry evaluation
will be against the correct ground truth.

### Issue 2: The curvature proxy formula needs clarification
The inconsistency between Eq.(16) and Remark 8 resolves beautifully into a
dual-regime theorem. The formula IS correct for both regimes; the paper just
fails to distinguish them. This is fixable with one Proposition.

### Missed Insight (Snn46, most important): κ̂ = isometry quality metric
The curvature proxy simultaneously measures curvature AND isometric calibration.
$\hat{\kappa} \to 2\sqrt{K}$ if and only if the decoder is isometrically
calibrated. This makes the experimental comparison much stronger:
the deviation of $\hat{\kappa}$ from $2\sqrt{K}$ is a QUANTITATIVE MEASURE
of remaining isometric distortion, not just a curvature accuracy test.

---

## Estimated Revision Timeline

| Task | Effort | Priority |
|---|---|---|
| Clifford torus experiment (data + code + rerun) | 2-3 days | Critical |
| Curvature proxy theorem (Snn46 Proposition 4) | 1 day | Critical |
| Lemma 1 proof fix | 2 hours | High |
| Fixed-point existence proposition | 2 hours | High |
| A4' capacity lemma | 1 day | High |
| Topological obstruction theorem | 1 day | High |
| Minor fixes (T3.3, T3.4, T4.*) | 1 day | Moderate |
| **Total** | **~8-10 days** | |

With these revisions the paper would be a strong NeurIPS contribution (all four
reviewers agree the core theory is sound; the issues are correctable).
