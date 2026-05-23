Let me re-read the full document one more time to make sure I catch every occurrence before writing the migration guide.Here is the complete migration guide.

---

## Notation Migration Guide

### Convention statement (to be added at the top of the document)

> **Notational conventions.** Superscript $(p)$ denotes a quantity that depends on $p$ and converges as $p\to\infty$; no superscript denotes the limit. Hat $\hat{\cdot}$ denotes a finite-$n$ sample quantity converging as $n\to\infty$; no hat denotes the population limit. Subscripts $B$, $b$, $F$ identify the matrix whose columns form the Gram.

---

### Table 1 — Symbol renames

| Old symbol                                        | New symbol                                        | What it is                                             | Notes                                                                                       |
| ------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| $d_j$                                             | $\lambda_j$                                       | population spike eigenvalue $c_j\sigma_j^2$            | frees $d$ from eigenvalue duty                                                              |
| $\rho_j$                                          | $\hat\lambda_j$                                   | eigenvalue of $\hat M$ at finite $n$                   | hat signals finite-$n$ sample quantity                                                      |
| $\tau_j$                                          | $\hat\lambda_j + \delta^2/n$                      | eigenvalue of $W_\infty$                               | eliminate symbol; write inline                                                              |
| $D$                                               | *(drop)*                                          | same as $M$ when $G^\infty_b = I_k$                    | redundant alias                                                                             |
| $\hat D$                                          | *(drop)*                                          | same as $\hat M$ when $G^\infty_b = I_k$               | redundant alias                                                                             |
| $\Gamma_p$                                        | $G^{(p)}_B$                                       | unnormalized loading Gram $B^\top B/p$ at finite $p$   | aligns with paper's $G$ family                                                              |
| $\Gamma_B$                                        | $G^\infty_B$                                      | limit of $G^{(p)}_B$                                   | $\infty$ superscript parallels $G^\infty_b$                                                 |
| $G(p)$                                            | $G^{(p)}_b$                                       | normalized loading Gram $b(p)^\top b(p)$ at finite $p$ | lowercase $b$ = normalized columns                                                          |
| $G_\infty$                                        | $G^\infty_b$                                      | limit of $G^{(p)}_b$                                   |                                                                                             |
| $F^\top F/n$ *(inline)*                           | $G^{(n)}_F$                                       | factor Gram at finite $n$                              | named for clarity; superscript $(n)$ since $n$ is the index                                 |
| $\Sigma_F$                                        | $\Sigma_F$                                        | population factor covariance, limit of $G^{(n)}_F$     | keep; standard notation                                                                     |
| $W_\infty$                                        | $W$                                               | limit of $W^{(p)}$                                     | no decoration = limit object                                                                |
| $\chi_{p,j}$                                      | $v^{(p)}_j$                                       | $j$-th eigenvector of $W^{(p)}$                        | makes convergence $v^{(p)}_j \to v_j$ explicit                                              |
| $s_{p,j}$                                         | $s^{(p)}_j$                                       | $j$-th singular value of $Y/\sqrt n$                   | consistent superscript convention                                                           |
| $a_j^\infty$                                      | $a_j$                                             | $\Gamma_B$-coordinate of population direction          | no decoration = limit                                                                       |
| $g_j^\infty$                                      | $g_j$                                             | $\Gamma_B$-coordinate of sample direction, limit       | no decoration = limit                                                                       |
| $b_j(p)$, $b(p)$                                  | $\beta_j$, $B$                                    | loading column / matrix                                | consolidate to one name                                                                     |
| $P$                                               | *(inline: $C^{1/2}$)*                             | square root factor of $G^\infty_B$                     | appears only twice; not worth a name                                                        |
| $\mathrm{SNR}_j$ (pop., $= nd_j/\delta^2$)        | $\mathrm{SNR}_j$                                  | population SNR $= n\lambda_j/\delta^2$                 | now unambiguous                                                                             |
| $\mathrm{SNR}_j$ (realized, $= n\rho_j/\delta^2$) | $\widehat{\mathrm{SNR}}_j$                        | realized SNR $= n\hat\lambda_j/\delta^2$               | hat distinguishes from population                                                           |
| $\sqrt{1-\text{floor}_j}$                         | $\sqrt{n\hat\lambda_j/(n\hat\lambda_j+\delta^2)}$ | shrinkage factor                                       | no prose inside math                                                                        |
| $\lambda_j(W_\infty)$ *(table header)*            | $\lambda_j(W)$                                    | eigenvalue of $W$                                      | just a notational use of $\lambda$, not a new symbol; consistent after rename of $W_\infty$ |

---

### Table 2 — Affected locations, old text → new text

| Location                | Old                                                                           | New                                                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| §1.1 parameter table    | $d_j = c_j\sigma_j^2$                                                         | $\lambda_j = c_j\sigma_j^2$                                                                                             |
| §1.1 parameter table    | $\mathrm{SNR}_j = nd_j/\delta^2$                                              | $\mathrm{SNR}_j = n\lambda_j/\delta^2$                                                                                  |
| §1.1 Assumption 3 check | $d_1 > d_2 > d_3$                                                             | $\lambda_1 > \lambda_2 > \lambda_3$                                                                                     |
| §1.1 prose              | *population* $\mathrm{SNR} = nd_j/\delta^2$                                   | $\mathrm{SNR}_j = n\lambda_j/\delta^2$                                                                                  |
| §1.1 prose              | *realized* $\mathrm{SNR} = n\rho_j/\delta^2$                                  | $\widehat{\mathrm{SNR}}_j = n\hat\lambda_j/\delta^2$                                                                    |
| §1.1 prose              | $\rho_j$ is the eigenvalue of the finite-$n$ matrix $\hat D$                  | $\hat\lambda_j$ is the eigenvalue of $\hat M$                                                                           |
| §1.2 prose              | $G(p) = b(p)^\top b(p) \approx I_k$                                           | $G^{(p)}_b = b(p)^\top b(p) \approx I_k$                                                                                |
| §1.3 prose              | $G_\infty = I_k$                                                              | $G^\infty_b = I_k$                                                                                                      |
| §1.3 prose              | $\Lambda_G = I_k$, $\Gamma_B = C$                                             | $\Lambda_G = I_k$, $G^\infty_B = C$                                                                                     |
| §1.3 display            | $\hat M = \hat D = C^{1/2}(F^\top F/n)C^{1/2}$                                | $\hat M = C^{1/2}G^{(n)}_F C^{1/2}$                                                                                     |
| §1.3 display            | $M = D = C^{1/2}\Sigma_F C^{1/2}$                                             | $M = C^{1/2}\Sigma_F C^{1/2}$                                                                                           |
| §1.3 table header       | $\rho_j$                                                                      | $\hat\lambda_j$                                                                                                         |
| §1.3 table header       | $d_j$ (pop. limit)                                                            | $\lambda_j$ (pop. limit)                                                                                                |
| §1.3 table header       | $\mathrm{SNR}_j = n\rho_j/\delta^2$                                           | $\widehat{\mathrm{SNR}}_j = n\hat\lambda_j/\delta^2$                                                                    |
| §1.3 section heading    | "Realized $\hat D$ eigenvalues"                                               | "Realized $\hat M$ eigenvalues"                                                                                         |
| §2 prose (B.3.1)        | $n\rho_3/\delta^2 \approx 0.41$                                               | $n\hat\lambda_3/\delta^2 \approx 0.41$                                                                                  |
| §3 term (A)             | $B^\top B/p = \Gamma_p \to \Gamma_B = C$                                      | $B^\top B/p = G^{(p)}_B \to G^\infty_B = C$                                                                             |
| §3 term (B) prose       | $b_j(p)$                                                                      | $\beta_j$                                                                                                               |
| §3 display (9)          | $W_\infty := F\Gamma_B F^\top/n + \ldots$                                     | $W := F G^\infty_B F^\top/n + \ldots$                                                                                   |
| §3 "in our example"     | $\Gamma_B = C = \mathrm{diag}(\ldots)$                                        | $G^\infty_B = C = \mathrm{diag}(\ldots)$                                                                                |
| §3 "in our example"     | $W_\infty = \ldots$                                                           | $W = \ldots$                                                                                                            |
| §3 display              | $\|W^{(500)} - W_\infty\|_\mathrm{op}$                                        | $\|W^{(500)} - W\|_\mathrm{op}$                                                                                         |
| §3 figure caption       | $W^{(p)} - W_\infty$                                                          | $W^{(p)} - W$                                                                                                           |
| §4 section heading      | "Eigenstructure of $W_\infty$"                                                | "Eigenstructure of $W$"                                                                                                 |
| §4 general arg.         | $\Gamma_B = PP^\top$ with $P = C^{1/2}Q\Lambda_G^{1/2}$                       | $G^\infty_B = PP^\top$ with $P = C^{1/2}Q\Lambda_G^{1/2}$                                                               |
| §4 general arg.         | $W_\infty - (\delta^2/n)I_n = (FP)(FP)^\top/n$                                | $W - (\delta^2/n)I_n = (FC^{1/2})(FC^{1/2})^\top/n$                                                                     |
| §4 display              | $(FP)^\top(FP)/n = P^\top(F^\top F/n)P = \hat M$                              | $(FC^{1/2})^\top(FC^{1/2})/n = C^{1/2}G^{(n)}_F C^{1/2} = \hat M$                                                       |
| §4 general arg.         | top-$k$ eigenvalues of $W_\infty$ are $\tau_j = \rho_j + \delta^2/n$          | top-$k$ eigenvalues of $W$ are $\hat\lambda_j + \delta^2/n$                                                             |
| §4 general arg.         | gap $\rho_k > 0$                                                              | gap $\hat\lambda_k > 0$                                                                                                 |
| §4 eigenvector display  | $v_j = FC^{1/2}\hat w_j/\sqrt{n\rho_j}$                                       | $v_j = FC^{1/2}\hat w_j/\sqrt{n\hat\lambda_j}$                                                                          |
| §4 table headers        | $\rho_j$, $\tau_j = \rho_j + \delta^2/n$, $\lambda_j(W_\infty)$               | $\hat\lambda_j$, $\hat\lambda_j + \delta^2/n$, $\lambda_j(W)$                                                           |
| §4 prose                | $\delta^2/n = 1/60$                                                           | unchanged                                                                                                               |
| §4 figure caption       | $\tau_3$, $\rho_3$                                                            | $\hat\lambda_3 + \delta^2/n$, $\hat\lambda_3$                                                                           |
| §5 general arg.         | $W^{(p)} \to W_\infty$                                                        | $W^{(p)} \to W$                                                                                                         |
| §5 display (11)         | $s_{p,j}^2/p \to \tau_j$, $\chi_{p,j} \to v_j$                                | $s^{(p)2}_j/p \to \hat\lambda_j + \delta^2/n$, $v^{(p)}_j \to v_j$                                                      |
| §5 prose                | $s_{p,j}$ singular values, $\chi_{p,j}$ eigenvectors                          | $s^{(p)}_j$, $v^{(p)}_j$                                                                                                |
| §5 prose                | gap $\rho_k > 0$                                                              | gap $\hat\lambda_k > 0$                                                                                                 |
| §5 table                | $s_{p,j}^2/p$, $\tau_j$, $\|\cos\angle(\chi_{p,j},v_j)\|$                     | $s^{(p)2}_j/p$, $\hat\lambda_j+\delta^2/n$, $\|\cos\angle(v^{(p)}_j,v_j)\|$                                             |
| §5 prose                | spectral gap $\rho_3 = 0.00679$                                               | $\hat\lambda_3 = 0.00679$                                                                                               |
| §5 prose                | $W^{(p)}$ closer to $W_\infty$                                                | $W^{(p)}$ closer to $W$                                                                                                 |
| §6 section heading      | "$\Gamma_B$-coordinate framework"                                             | "$G^\infty_B$-coordinate framework"                                                                                     |
| §6 display              | $\Phi_p^\top\Phi_p = B^\top B/p = \Gamma_p \to \Gamma_B$                      | $\Phi_p^\top\Phi_p = G^{(p)}_B \to G^\infty_B$                                                                          |
| §6 prose                | "$\Gamma_B$-inner product $x^\top\Gamma_B y$"                                 | "$G^\infty_B$-inner product $x^\top G^\infty_B y$"                                                                      |
| §6 prose                | "SVD of $b(p)$"                                                               | "SVD of $B$"                                                                                                            |
| §6 prose                | $a_j^{(p)} \to a_j^\infty$                                                    | $a_j^{(p)} \to a_j$                                                                                                     |
| §6 display              | $a_j^\infty = C^{-1/2}e_j$                                                    | $a_j = C^{-1/2}e_j$                                                                                                     |
| §6 display              | $g_j^{(p)} \to g_j^\infty := F^\top v_j/\sqrt{n\rho_j+\delta^2}$              | $g_j^{(p)} \to g_j := F^\top v_j/\sqrt{n\hat\lambda_j+\delta^2}$                                                        |
| §6 display (12)         | $g_j^\infty = \sqrt{n\rho_j/(n\rho_j+\delta^2)}\cdot C^{-1/2}\hat w_j$        | $g_j = \sqrt{n\hat\lambda_j/(n\hat\lambda_j+\delta^2)}\cdot C^{-1/2}\hat w_j$                                           |
| §6 table                | $g_j^\infty$, $a_j^\infty$                                                    | $g_j$, $a_j$                                                                                                            |
| §6 prose                | $\hat D$ is not exactly diagonal                                              | $\hat M$ is not exactly diagonal                                                                                        |
| §6 prose                | $\sqrt{n\rho_j/(n\rho_j+\delta^2)}$ … $\sqrt{1-\text{floor}_j}$               | $\sqrt{n\hat\lambda_j/(n\hat\lambda_j+\delta^2)}$ … $\sqrt{\delta^2/(n\hat\lambda_j+\delta^2)}$ *(drop informal alias)* |
| §7 section heading      | "via $\Gamma_B$ inner products"                                               | "via $G^\infty_B$ inner products"                                                                                       |
| §7 prose                | $g_j^{(p)}\to g_j^\infty$, $a_j^{(p)}\to a_j^\infty$, $\Gamma_p\to\Gamma_B$   | $g_j^{(p)}\to g_j$, $a_j^{(p)}\to a_j$, $G^{(p)}_B\to G^\infty_B$                                                       |
| §7.1 display            | $(g_j^{(p)})^\top\Gamma_p g_j^{(p)} \to (g_j^\infty)^\top\Gamma_B g_j^\infty$ | $(g_j^{(p)})^\top G^{(p)}_B g_j^{(p)} \to g_j^\top G^\infty_B g_j$                                                      |
| §7.1 display            | $n\rho_j/(n\rho_j+\delta^2)$ throughout                                       | $n\hat\lambda_j/(n\hat\lambda_j+\delta^2)$                                                                              |
| §7.1 display (13)       | $\delta^2/(n\rho_j+\delta^2)$                                                 | $\delta^2/(n\hat\lambda_j+\delta^2)$                                                                                    |
| §7.2 display            | $(g_j^{(p)})^\top\Gamma_p a_j^{(p)} \to (g_j^\infty)^\top\Gamma_B a_j^\infty$ | $(g_j^{(p)})^\top G^{(p)}_B a_j^{(p)} \to g_j^\top G^\infty_B a_j$                                                      |
| §7.2 display            | $\sqrt{n\rho_j/(n\rho_j+\delta^2)}$                                           | $\sqrt{n\hat\lambda_j/(n\hat\lambda_j+\delta^2)}$                                                                       |
| §7.3 display            | $n\rho_j/(n\rho_j+\delta^2)$ (twice)                                          | $n\hat\lambda_j/(n\hat\lambda_j+\delta^2)$                                                                              |
| §7 table                | $(g_j^\infty)^\top\Gamma_B g_j^\infty$, $n\rho_j/(n\rho_j+\delta^2)$          | $g_j^\top G^\infty_B g_j$, $n\hat\lambda_j/(n\hat\lambda_j+\delta^2)$                                                   |
| §7 table                | Floor $= \delta^2/(n\rho_j+\delta^2)$                                         | Floor $= \delta^2/(n\hat\lambda_j+\delta^2)$                                                                            |
| §7 prose                | $\hat D$ is close to diagonal                                                 | $\hat M$ is close to diagonal                                                                                           |
| §8 display              | $\delta^2/(n\rho_j+\delta^2)$, $n\rho_j/(n\rho_j+\delta^2)$                   | $\delta^2/(n\hat\lambda_j+\delta^2)$, $n\hat\lambda_j/(n\hat\lambda_j+\delta^2)$                                        |
| §8 prose                | "$\Gamma_B$-coordinate framework … $\Gamma_p\to\Gamma_B$"                     | "$G^\infty_B$-coordinate framework … $G^{(p)}_B\to G^\infty_B$"                                                         |
| §8 prose                | "SVD basis $U(p)$"                                                            | unchanged                                                                                                               |
| §9 prose                | $\|W^{(p)}-W_\infty\|*\mathrm{op}\to 0$, $\chi*{p,j}\to v_j$                  | $\|W^{(p)}-W\|_\mathrm{op}\to 0$, $v^{(p)}_j\to v_j$                                                                    |
| §10 display             | $n\rho_j+\delta^2$ (twice)                                                    | $n\hat\lambda_j+\delta^2$                                                                                               |
| §10 prose               | $\sum_j 1/(1+\mathrm{SNR}_j)$                                                 | unchanged (uses population $\mathrm{SNR}_j$, now unambiguous)                                                           |
| §11 summary table       | "$n\times n$ limit $W_\infty$"                                                | "$n\times n$ limit $W$"                                                                                                 |
| §11 summary table       | "$\Gamma_B$ coordinates", "$\Gamma_B$ bilinear forms"                         | "$G^\infty_B$ coordinates", "$G^\infty_B$ bilinear forms"                                                               |
| §11 prose               | "$\Gamma_B$-coordinate framework … $\Gamma_p\to\Gamma_B$"                     | "$G^\infty_B$-coordinate framework … $G^{(p)}_B\to G^\infty_B$"                                                         |
| §11 prose               | "SVD basis of $b(p)$"                                                         | "SVD basis of $B$"                                                                                                      |

---

### Items left unchanged

$k$, $p$, $n$, $\delta^2$, $c_j$, $\sigma_j^2$, $B$, $F$, $Z$, $Y$, $h_j$, $\bar b_j$, $\mathcal{B}$, $\Pi_B$, $M$, $\hat M$, $w_j$, $\hat w_j$, $v_j$, $a_j^{(p)}$, $g_j^{(p)}$, $\Phi_p$, $Q$, $\Lambda_G$, $C$, $W^{(p)}$, $\Sigma_F$, $e_j$, $d_{\mathrm{Gr}}$, $U(p)$.
