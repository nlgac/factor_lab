## Why Stiefel distances are erratic while Grassmannian distances are monotone

The short answer: **the Stiefel canonical metric is sensitive to in-plane SO(k) rotation of the basis vectors within the subspace, and that rotation is uncontrolled by p. The Grassmann metric is blind to that rotation entirely.**

---

### What each metric actually measures

From the documentation, `grassmann_distance(U1, U2)` computes the principal angles between the two k-planes via SVD of U1ᵀU2. It is completely invariant to how the basis vectors inside each plane are oriented — two frames spanning the same subspace have Grassmann distance 0 regardless of the SO(k) rotation between them.

`stiefel_canonical_distance(U1, U2)` uses the 2k×2k block rotation matrix G and the Schur decomp to extract both components of the geodesic:

```
d_S² = ½‖A11‖_F² + ‖A21‖_F²
```

A11 is the **vertical** (in-plane SO(k) rotation) component; A21 is the **horizontal** (subspace-changing) component. The Stiefel distance penalizes both.

---

### Why Grassmann distances decrease monotonically

The Grassmann targets are generated with A11 = 0 — pure horizontal motion. The measured Grassmann distance from B^S to those targets depends only on how close the *subspace* of B^S is to the *subspace* of B^GT, plus the subspace geometry of the targets themselves.

As p increases, you're including more assets, which gives the SVD more information about the factor subspace. Standard concentration-of-measure arguments apply: the sample frame B^S converges toward B^GT in subspace angle, and the distribution of Grassmann distances tightens monotonically. You can see this directly in the data — both the mean and std decrease smoothly across p = 100, 500, 1000, 3000, 5000, 10000.

---

### Why Stiefel distances are erratic

There are two compounding reasons:

**1. The SVD has an uncontrolled SO(k) rotation at every p-slice.**

When you compute the top-k eigenvectors of the sample covariance, the result is a valid orthonormal frame, but the *orientation* of that frame within the k-dimensional subspace is not uniquely determined. The SVD will produce some particular rotation of the eigenvectors, but which rotation depends on the eigenvalue structure at that specific p. There is no mechanism that pins it to the orientation B^GT uses.

The Stiefel distance from B^S to a target placed around B^GT sees both the subspace error (same as Grassmann) *and* this SO(k) misalignment. The misalignment is not controlled by p — it's controlled by the relative separations of the three factor eigenvalues in the sliced covariance, which change in a complex way as you add more assets.

**2. The sliced model changes non-monotonically.**

The documentation explicitly notes this:

> *Using one population model and slicing to each p is an explicit design choice. It conflates subset-selection effects with genuine dimension effects.*

As p grows, you're not adding observations to better estimate a fixed model — you're changing the model by including more assets with different loading profiles. The factor variance ratios, the SNR of each factor relative to idiosyncratic noise, and the eigenvector orientations all shift when you go from p=1000 to p=3000 to p=5000. Each of these changes produces a different SO(k) rotation in B^S relative to B^GT.

The Grassmann metric is immune to this because it doesn't care which eigenvectors the SVD chose — only which subspace they span. The Stiefel metric is fully exposed to it.

---

### Reading this in the numbers

For radius=0.1, looking at `sample-truth` means (which directly measure the estimated vs. ground-truth distance):

| p     | Grassmann mean | Stiefel mean |
| ----- | -------------- | ------------ |
| 100   | 1.608          | 2.447        |
| 500   | 1.394          | 3.299        |
| 1000  | 1.272          | 2.074        |
| 3000  | 1.209          | 3.319        |
| 5000  | 1.192          | 3.376        |
| 10000 | 1.185          | 2.748        |

The Grassmann sample-truth mean decreases smoothly and monotonically. The Stiefel sample-truth mean jumps from 2.45 → 3.30 → 2.07 → 3.32 → 3.38 → 2.75 — none of which is a function of p alone. This is the in-plane SO(3) rotation being picked up differently at each p-slice, with no inherent ordering.

---

### Implication for the research

This is actually important for interpreting the dispersion bias question. The Grassmann distances give you a clean signal about subspace estimation quality as a function of dimensionality. The Stiefel distances are mixing that signal with SO(k) orientation noise that is essentially an artifact of the slicing design. If you want to use Stiefel distances to study bias, you'd either need to (a) align frames before measuring (e.g., project out the SO(k) ambiguity), or (b) build independent models at each p rather than slicing a single one.
