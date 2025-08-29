
# Theoretical Foundations of MDL in `mdllosstorch`

## 1. Introduction

The `mdllosstorch` package implements **Minimum Description Length (MDL)** as a unified loss criterion for neural models. This perspective treats learning as data compression: the best model is the one that yields the shortest overall code length.

Conventional approaches use ad hoc regularizers (e.g., L1/L2 penalties, dropout, Bayesian priors). MDL subsumes these: it naturally penalizes model complexity by accounting for the bits required to encode parameters and residuals. No extra regularization is necessary.

To distinguish from the legacy "AIC" (Akaike Information Criterion), we refer to this principle as the **Algorithmic Loss/Information Criterion (ALIC)**.

---

## 2. The Algorithmic Information Criterion (ALIC)

The ALIC formalizes model selection as:

```
total bits = parameter bits + residual bits
```

- **Parameters** are encoded using probabilistic models (e.g., Student-t with discretization).  
- **Residuals** are transformed (Box–Cox or Yeo–Johnson) to approximate Gaussianity, then encoded with variance-based coding.  
- **Discretization** ensures positivity and prevents degenerate encodings.

This approximation makes MDL tractable as a differentiable loss function.

---

## 3. Guidelines for Proper Application

### Do’s
- Use ALIC as a **single unified loss**.  
- Treat it as both a **loss function** (for training candidate models) and a **criterion** (for comparing architectures).  
- Periodically verify using more computationally expensive techniques (see Section 4).

### Don’ts
- Do **not** add L1/L2/dropout or other penalties "on top" — that double-counts costs already accounted for by ALIC.  
- Do **not** compare models trained on different datasets — MDL only compares models of the same data.  
- Do **not** confuse the theoretical uncomputability of Kolmogorov complexity with the practical usefulness of relative MDL comparisons.

---

## 4. Verification Pathways

### Cheap approximation (default)
- Closed-form coding of parameters and residuals.  
- Differentiable and fast enough for training.

### Expensive verification
- Use **genetic algorithms** or other search heuristics to approximate empirical parameter/residual distributions.  
- Encode datasets using **arithmetic coding** under those distributions.  
- Compare archive sizes between competing models for a more principled (but slower) approximation.

This two-tier strategy balances practicality (training loss) and rigor (verification).

---

## 5. Epistemological Note

ALIC is inherently a form of **forensic epistemology**: it evaluates models by the evidence encoded in their data compressibility.

- Arguments about dataset bias matter less as datasets grow: forensic evidence dominates anecdotal bias.  
- ALIC reframes debates about "which data counts" by quantifying bias as part of the compression framework.  

---

## Appendix A: Fallacies & Responses

### Fallacy 1: *“You only count parameters + residuals, ignoring algorithmic complexity.”*  
**Response:** The algorithmic description length of the neural simulation engine is orders of magnitude smaller than the dataset + parameters. Differences between architectures are noise relative to ALIC.

---

### Fallacy 2: *“Exact dataset reproduction is the true MDL; approximations are invalid.”*  
**Response:** True in principle, but irrelevant for model training. ALIC is used to compare candidate models. Expensive arithmetic coding checks can approximate the true MDL for validation.

---

### Fallacy 3: *“Model selection is data selection.”*  
**Response:** ALIC compares models of the *same dataset*. Dataset bias matters, but its importance decreases with scale. ALIC formalizes what critics otherwise frame as "forensic epistemology" of data bias.

---

### Fallacy 4: *“Algorithmic information is uncomputable, so ALIC is meaningless.”*  
**Response:** Uncomputability only means we cannot prove global optimality. ALIC is still valid for *relative comparisons* among discovered models.

---

## 6. Conclusion

`mdllosstorch` operationalizes MDL as both a **loss function** and a **model selection criterion**. It unifies regularization, training, and evaluation under a single information-theoretic principle. By combining tractable approximations with optional expensive verification, it offers both practical usability and theoretical rigor.
