# Calmodulin–RS20: literature values for the calcium parameters

Literature review for the calcium-control parameters of the simulation (`Params` in `qtft/params.py`,
model in `docs/decisions.md`, section "Calcium control"). The bond between Qt and Ft stands for the
calmodulin–RS20 interaction: RS20 is the 20-residue peptide of the calmodulin-binding domain of
**smooth-muscle myosin light chain kinase** (smMLCK), the peptide used in the GCaMP calcium indicators.

The current defaults were order-of-magnitude estimates with no citation. This file collects measured
values, says what each one does and does not cover, and recommends parameters. Every number below is
from the paper named next to it; where no measurement exists for RS20 itself, that is stated.

Date of the review: 2026-10-26. Nothing in the code was changed.

---

## 1. Summary of the recommendations

| Parameter | Current default | Recommended | Basis |
|---|---|---|---|
| `ca_K_uM` | 0.5 µM | **0.5 µM** (keep), range 0.2–1 µM | Ca²⁺ affinity of CaM rises ~9-fold (C-lobe) with RS20 (Johnson 1996); CaM–M13 nanoparticle aggregation has EC50 = 1.4 µM (Atanasijevic 2006) |
| `ca_hill` | 3 | **3** (keep), range 2.8–4.3 | Hill coefficients 2.8–4.7 measured for CaM–RS20 indicators (Helassa 2015, 2016) |
| `k_on` (f = 1) | 1.86·10⁻⁶ ns⁻¹ ≈ 2·10⁷ M⁻¹s⁻¹ | **keep**, with the note that the peptide step itself is ~8.8·10⁸ M⁻¹s⁻¹ | Török & Trentham 1994; the simulation value is a coarse-grained particle–particle rate, not the peptide rate |
| `k_off_ca_per_s` | 10⁻³ s⁻¹ | **7.9·10⁻³ s⁻¹** (measured), keep 10⁻³ as the "tight" variant | Török & Trentham 1994: 0.0079 s⁻¹ for the Trp peptide; 0.38–0.65 s⁻¹ for the Tyr variant |
| `k_off_free_per_s` | 1.0 s⁻¹ | **keep 1 s⁻¹**, range 0.4–5 s⁻¹ | Set by Ca²⁺ dissociation from the complex, 0.1–2.9 s⁻¹ (Johnson 1996; Persechini 1996) |
| `ca_rest_uM`, `ca_peak_uM` | 0.1, 1.0 µM | **0.1, 0.5–0.8 µM** | Opto-CRAC reaches 500–800 nM (He 2015) |
| `ca_rise_s` | 5 s | **~30 s** (t½ 23–36 s) if optoCRAC-like | He 2015 |
| `ca_tau_s` | 20 s (runs: 15 s) | **~40 s** (t½ 25–35 s) | He 2015 |

The two parameters that should change on the evidence are the **calcium protocol timings** (`ca_rise_s`,
`ca_tau_s`, `ca_peak_uM`) and `k_off_ca_per_s`. The loading parameters `ca_K_uM` and `ca_hill` are
supported by the measurements and need no change.

**Decisions taken on 2026-10-26** (recorded here so the reasoning is not lost):

- `k_off_ca_per_s` stays at **10⁻³ s⁻¹**, not the measured 7.9·10⁻³ s⁻¹: Török & Trentham worked at
  100 µM Ca²⁺, far above the ≤ 1 µM of the simulated protocol, so their off-rate is not directly
  transferable. The choice also barely matters — at f(peak) = 0.89 the unloaded fraction contributes
  ~94 % of k_off either way (see section 3.2).
- The **0.098 calibration factor was dropped** from `control.ipynb`, which now uses the unscaled
  literature values 10⁻³ / 1 s⁻¹. That factor existed only to reproduce an artifact of the superseded
  `bond_break="longest"` method; with `"random"` k_off is the true per-bond rate. See
  `docs/decisions.md`, "Unbinding".
- `ca_tau_s` raised 15 → **20 s**, a step towards the measured optoCRAC decay (t½ 25–35 s ⇒ τ ≈ 36–50 s),
  without changing the protocol so much that it becomes incomparable with earlier runs.

---

## 2. The calcium-loading curve f(Ca) = Caⁿ/(Kⁿ + Caⁿ)

### 2.1 Hill coefficient n

Direct measurements exist, because GCaMP indicators are CaM–RS20 fusions and their calcium titrations
are exactly this curve.

| Construct | K_d (Ca²⁺) | Hill n | Temperature | Source |
|---|---|---|---|---|
| GCaMP3 (CaM + RS20) | 0.33 ± 0.01 µM | **3.2 ± 0.1** | 20 °C | Helassa 2015, Table 1 |
| GCaMP3_fast (RS-1 EF-3) | 2.8 ± 0.3 µM | 4.3 ± 0.2 | 20 °C | Helassa 2015 |
| GCaMP3_bright (EF-4) | 0.93 ± 0.01 µM | 4.7 ± 0.1 | 20 °C | Helassa 2015 |
| GCaMP6f | 0.22 µM | **2.8** | 20 °C | Helassa 2016, Table 1 |
| GCaMP6f | 0.09 µM | 3.4 | 37 °C | Helassa 2016 |
| GCaMP6f_u | 0.89 / 0.34 µM | 3.0 | 20 / 37 °C | Helassa 2016 |

**Conclusion:** n = 3 is right in the middle of the measured range (2.8–4.7). The current default needs
no change. A value of 4 would also be defensible (4 sites, two cooperative lobes).

**Caveat:** these are fluorescence titrations of the whole indicator, so n describes the Ca²⁺ dependence
of the complete CaM–peptide–fluorophore system, which is what the model needs, but the GCaMP
fluorophore attachment may shift it from the free CaM–RS20 complex.

### 2.2 Half-loading K

No published Ca²⁺ titration of a plain CaM–RS20 complex was found. Three lines of evidence bracket it:

1. **Target peptides raise the Ca²⁺ affinity of CaM.** Johnson et al. (1996) measured, at 10 °C:
   - C-lobe Ca²⁺ affinity **9-fold higher with RS-20**; 6.3-fold with a CaM kinase II peptide.
   - Ca²⁺ dissociation slows from 405 ± 75 s⁻¹ (free N-lobe) to 1.8–2.9 s⁻¹ with RS-20, M-13, CaM
     kinase II or MARCKS peptides; from 2.4 ± 0.2 s⁻¹ (free C-lobe) to 0.1–0.4 s⁻¹.
   - Ca²⁺ association: 1.6·10⁸ M⁻¹s⁻¹ (N-lobe), 2.3·10⁶ M⁻¹s⁻¹ (C-lobe).
2. **GCaMP indicators with RS20** have K_d = 0.09–0.33 µM for the wild-type-like variants and up to
   2.8 µM for weakened ones (Helassa 2015, 2016). These are below 0.5 µM.
3. **A system closer to ours**: Atanasijevic et al. (2006) built calcium-sensitive MRI agents from iron
   oxide nanoparticles carrying CaM and the MLCK peptide, which **aggregate when calcium binds** — the
   same physical mechanism as in this simulation, on 20–100 nm particles. Their aggregation titration
   gives **EC50 = 1.4 ± 0.1 µM Ca²⁺** (CaM:M13 at 3:1), 0.8 ± 0.3 µM for one variant and 10 ± 2 µM for a
   low-affinity variant. The paper also used RS20 sensors (3C:1R, 1C:1R) with similar behaviour but
   reports no separate EC50 for them.

**Conclusion:** the isolated complex sits near 0.1–0.3 µM, but the aggregation of particles carrying
many copies of the pair is half-maximal around 1 µM. The model's f is the *particle-level* switch, so a
value **between the two, 0.5 µM, is the right choice.** Keep the default. Range for a sensitivity
sweep: 0.2–1.0 µM.

**Caveat:** Johnson's 9-fold figure is for the C-lobe at 10 °C and low ionic strength. Peersen et al.
(1997) report that different targets tune the Ca²⁺ affinity differently, so the factor is
target-specific.

---

## 3. Binding and unbinding rates

### 3.1 The peptide step: Török & Trentham (1994)

This is the key kinetic paper. It measured smMLCK and derived 17-residue peptides binding to
fluorescently labelled calmodulin (TA-calmodulin, labelled at Lys75) at 100 µM CaCl₂, ionic strength
0.15 M, pH 7.0, **21 °C**. The mechanism is two-step: fast bimolecular association, then a slow
isomerisation.

| Quantity | Trp peptide | Tyr peptide (W→Y) | Whole smMLCK |
|---|---|---|---|
| k₊₁ (association) | 8.8·10⁸ M⁻¹s⁻¹ | 8.8·10⁸ M⁻¹s⁻¹ | 1.1·10⁸ M⁻¹s⁻¹ |
| k₋₁ (dissociation of the first complex) | — | 5.7 s⁻¹ | — |
| isomerisation k₊₂ / k₋₂ | (k₊₂+k₋₂) = 0.97 s⁻¹ | 0.38 / 0.65 s⁻¹ | (k₊₂+k₋₂) = 1.3 s⁻¹ |
| **overall off-rate** k₋₁k₋₂/(k₊₂+k₋₂) | **0.0079 s⁻¹** | — | 0.025–0.056 s⁻¹ |
| **K_d** | **0.011 nM** | 4.1 nM | 0.23–0.51 nM |

Helassa et al. (2015) quote the same 0.0075 s⁻¹ as the peptide dissociation rate from the
Ca²⁺·CaM·peptide complex.

**What this means for `k_off_ca_per_s`:** the measured value is **≈ 8·10⁻³ s⁻¹** (bond lifetime ~2 min),
not 10⁻³ s⁻¹. The current default is 8× too tight. Note the strong sequence sensitivity: the single
Trp→Tyr change raises K_d from 0.011 nM to 4.1 nM, a factor of 370. The fusion construct of the
experiment may therefore sit anywhere in that range.

**What this means for `k_on`:** the peptide associates at 8.8·10⁸ M⁻¹s⁻¹, close to the diffusion limit
for a small peptide. The simulation's `k_on` is **not** this number: it is the rate at which two
*particles already in contact* form a bond, which folds in the number and orientation of the surface
sites and the slower diffusion of a 55 nm cage. The current value (≈ 2·10⁷ M⁻¹s⁻¹ apparent k_a) is a
reasonable coarse-grained choice and is calibrated against experiment, not literature. The literature
value is an **upper bound** on what one pair of sites can do.

### 3.2 Unbinding after calcium removal: `k_off_free_per_s`

When calcium leaves, the complex falls apart. Because the peptide off-rate from the Ca²⁺-loaded
complex (0.008 s⁻¹) is far slower than calcium dissociation, the **release of the peptide is limited by
the Ca²⁺ off-rate**, which Helassa (2015, 2016) also assume in their kinetic model.

Measured Ca²⁺ dissociation rates from CaM–target complexes:

| Complex | Rate | Sites | Source |
|---|---|---|---|
| CaM alone, N-lobe | 405 ± 75 s⁻¹ | — | Johnson 1996 (10 °C) |
| CaM alone, C-lobe | 2.4 ± 0.2 s⁻¹ | — | Johnson 1996 (10 °C) |
| CaM + RS-20 / M-13 / CaMKII / MARCKS, N-lobe | **1.8–2.9 s⁻¹** | — | Johnson 1996 |
| CaM + target peptides, C-lobe | **0.1–0.4 s⁻¹** | — | Johnson 1996 |
| CaM–skMLCK peptide | 0.15 s⁻¹ (3 sites), 2 s⁻¹ (1 site) | 4 | Persechini 1996 |
| CaM–skMLCK (whole kinase) | 1.6 s⁻¹ (3 sites), > 1000 s⁻¹ (1 site) | 4 | Persechini 1996 |

**Conclusion:** the bond falls apart on the time scale of the **slowest** Ca²⁺ site, 0.1–0.4 s⁻¹, up to a
few s⁻¹. `k_off_free_per_s` = 1 s⁻¹ sits inside this range; keep it. Range for a sweep: 0.4–5 s⁻¹.

**Caveat:** the measurements are at 10 °C; rates at 37 °C are several-fold faster (Helassa 2015 finds
3–4× from 20 to 37 °C for the indicator). A value at the upper end of the range is defensible for
37 °C.

### 3.3 Why the model's two-rate scheme is a simplification

The model uses k_off(t) = f·k_off,Ca + (1−f)·k_off,free, i.e. the bond is either "loaded" or "free".
The literature says the real path is a sequence: Ca²⁺ leaves one lobe (fast), then the other (slow),
then the peptide. There is also no CaM–RS20–Ca₂ intermediate (Hill 2000: only 1:1:4 CaM/RS20/Ca
complexes, no Ca₂ species), so the all-or-nothing picture is not unreasonable. The mean-field f with
n = 3 captures the cooperativity that makes the switch sharp.

**One point the model ignores:** RS20 also binds calcium-free calmodulin. Tsvetkov et al. (1999)
measured, by isothermal titration calorimetry at 25 °C, pH 7.5:

| Condition | K_a (apo-CaM–RS20) | ≈ K_d |
|---|---|---|
| no salt | 1.0·10⁶ M⁻¹ | ~1 µM |
| 100 mM NaCl | 4.2·10³ M⁻¹ | ~240 µM |
| 150 mM KCl | 1.2·10³ M⁻¹ | ~830 µM |
| 150 mM KCl + 5 mM MgCl₂ | 3.6·10³ M⁻¹ | ~280 µM |

The apo complex uses only the C-lobe of calmodulin and is driven by electrostatics, so **physiological
salt nearly abolishes it** (a ~1000-fold drop from no salt). Hill et al. (2000) confirmed the apo
complex by mass spectrometry at low ionic strength. **For the simulation this means: at cellular ionic
strength the calcium-free bond is weak and the model's "unbinding to free particles" is justified.**
Residual binding is not zero, though, which is a reason why `k_off_free_per_s` should not be pushed far
above ~1 s⁻¹.

---

## 4. The calcium protocol

The experiment uses optoCRAC (LOV2–STIM1 and ORAI1), for which He et al. (2015) report measured
kinetics in non-excitable cells (470 nm, 40 µW/mm²):

| Quantity | Measured | Current model |
|---|---|---|
| LOVSoc translocation to the membrane | t½,on = 6.8 ± 2.3 s | — |
| LOVSoc return after light off | t½,off = 28.7 ± 6.5 s | — |
| **Cytosolic Ca²⁺ rise** | **t½ = 23–36 s** | `ca_rise_s` = 5 s to 99 %, i.e. t½ ≈ 0.75 s |
| **Cytosolic Ca²⁺ decay after light off** | **t½ ≈ 25–35 s** | `ca_tau_s` = 15–20 s, i.e. t½ ≈ 10–14 s |
| **Peak cytosolic Ca²⁺** | **≈ 500–800 nM** after 1 min | `ca_peak_uM` = 1.0 µM |

**Conclusion:** the model's calcium rises and falls considerably faster than measured optoCRAC.
Converted to the model's parameters:
- `ca_rise_s` (time to 99 %, = t½·ln100/ln2 ≈ 6.6·t½) would be **≈ 150–240 s** for a t½ of 23–36 s.
  That is longer than the current 30 s influx phase, so a protocol of this kind would need a longer
  influx or a different light intensity.
- `ca_tau_s` = t½/ln2 ≈ **36–50 s** instead of 15–20 s.
- `ca_peak_uM` = **0.5–0.8** instead of 1.0. With K = 0.5 µM and n = 3 this changes f(peak) from 0.89 to
  0.50–0.80, a noticeable drop in the binding rate.

**Caveat:** He et al. measured HeLa, HEK293 and immune cells with a particular light dose. The rise
time depends strongly on intensity, and the local calcium at the membrane is higher than the bulk
value. The right source is a calcium trace measured in the group's own cells and setup; these numbers
are the fallback and show the order of magnitude.

---

## 5. What is still missing

1. **No Ca²⁺ titration of a plain CaM–RS20 complex.** All Hill coefficients and K values above come
   from GCaMP fusions or from aggregating nanoparticles.
2. **No kinetics for the RS20 variant actually used in the experiment.** The Trp→Tyr example shows a
   370-fold affinity change from one residue, so the construct's sequence matters more than any
   literature average.
3. **No measurement on Qt/Ft cages.** The nearest system is Atanasijevic's 20–100 nm nanoparticles,
   which aggregate reversibly with an EC50 of 1.4 µM, disaggregate "within a few seconds" after EDTA
   and take tens of minutes to aggregate fully. That timescale asymmetry (fast disassembly, slow
   assembly) is worth comparing with the simulation, which currently assembles in ~10 s and
   disassembles over ~90 s.
4. **Avidity.** A pair of cages is held by many bonds, so the effective pair affinity is far higher
   than one CaM–RS20 bond. The model captures this through the bond count (`maxbond_qt` = 60,
   `maxbond_ft` = 10), not through k_off, which is the right place for it.

---

## 6. Sources

Measured values used above:

1. **Török K, Trentham DR (1994).** "Mechanism of 2-chloro-(ε-amino-Lys75)-[6-[4-(N,N-diethylamino)
   phenyl]-1,3,5-triazin-4-yl]calmodulin interactions with smooth muscle myosin light chain kinase and
   derived peptides." *Biochemistry* 33(43), 12807–12820. PMID 7947686.
   → peptide association 8.8·10⁸ M⁻¹s⁻¹, overall off-rate 0.0079 s⁻¹, K_d 0.011 nM (Trp peptide),
   4.1 nM (Tyr peptide), 0.23–0.51 nM (whole smMLCK); 21 °C, pH 7.0, I = 0.15 M.
2. **Johnson JD, Snyder C, Walsh M, Flynn M (1996).** "Effects of myosin light chain kinase and
   peptides on Ca²⁺ exchange with the N- and C-terminal Ca²⁺ binding sites of calmodulin."
   *J Biol Chem* 271(2), 761–767. PMID 8557684.
   → Ca²⁺ off-rates with RS-20 and other peptides, 9-fold C-lobe affinity increase with RS-20,
   Ca²⁺ on-rates 1.6·10⁸ (N) and 2.3·10⁶ M⁻¹s⁻¹ (C); 10 °C.
3. **Persechini A, White HD, Gansz KJ (1996).** "Different mechanisms for Ca²⁺ dissociation from
   complexes of calmodulin with nitric oxide synthase or myosin light chain kinase."
   *J Biol Chem* 271(1), 62–67. PMID 8550626.
   → CaM–skMLCK peptide: 0.15 s⁻¹ (3 sites) and 2 s⁻¹ (1 site).
4. **Helassa N, Zhang X-H, Conte I, Scaringi J, Esposito E, Bradley J, Carter T, Ogden D, Morad M,
   Török K (2015).** "Fast-Response Calmodulin-Based Fluorescent Indicators Reveal Rapid Intracellular
   Calcium Dynamics." *Sci Rep* 5, 15978. <https://pmc.ncbi.nlm.nih.gov/articles/PMC4630588>
   → GCaMP3 (CaM + RS20, explicitly "the smooth muscle myosin light chain kinase derived calmodulin
   binding peptide (RS20)"): K_d 0.33 µM, n = 3.2; variants up to n = 4.7; peptide release modelled as
   limited by Ca²⁺ dissociation.
5. **Helassa N, Podor B, Fine A, Török K (2016).** "Design and mechanistic insight into ultrafast
   calcium indicators for monitoring intracellular calcium dynamics." *Sci Rep* 6, 38276.
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC5138832>
   → GCaMP6f: K_d 0.22 µM (20 °C) / 0.09 µM (37 °C), n = 2.8–3.4.
6. **Tsvetkov PO, Protasevich II, Gilli R, Lafitte D, Lobachov VM, Haiech J, Briand C, Makarov AA
   (1999).** "Apocalmodulin Binds to the Myosin Light Chain Kinase Calmodulin Target Site."
   *J Biol Chem* 274(26), 18161–18164.
   <https://web.math.princeton.edu/~sswang/GECI/tsvetkov_makarov99_jbc_apoCaM-MLCK-binding.pdf>
   → apo-CaM–RS20: K_a 1.0·10⁶ M⁻¹ (no salt) down to 1.2·10³ M⁻¹ (150 mM KCl); C-lobe only; 25 °C.
7. **Hill TJ, Lafitte D, Wallace JI, Cooper HJ, Tsvetkov PO, Derrick PJ (2000).** "Calmodulin–Peptide
   Interactions: Apocalmodulin Binding to the Myosin Light Chain Kinase Target-Site."
   *Biochemistry* 39(24), 7284–7290.
   <https://web.math.princeton.edu/~sswang/GECI/hill_derrick00_biochemistry_apoCaM-MLCK-binding.pdf>
   → 1:1:4 CaM/RS20/Ca stoichiometry, no CaM–RS20–Ca₂ species (cooperative lobes); K_d 10⁻⁶–10⁻⁷ M
   quoted for apo peptide binding.
8. **Atanasijevic T, Shusteff M, Fam P, Jasanoff A (2006).** "Calcium-sensitive MRI contrast agents
   based on superparamagnetic iron oxide nanoparticles and calmodulin." *PNAS* 103(40), 14707–14712.
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC1595416>
   → calcium-triggered, reversible aggregation of 20–100 nm particles carrying CaM and M13 or RS20;
   EC50 = 1.4 ± 0.1 µM (3C:1M), 0.8 ± 0.3 µM and 10 ± 2 µM for variants; disaggregation within seconds
   after EDTA, aggregation over tens of minutes.
9. **He L, Zhang Y, Ma G, Tan P, Li Z, Zang S, Wu X, Jing J, Fang S, Zhou L, Wang Y, Huang Y, Hogan PG,
   Han G, Zhou Y (2015).** "Near-infrared photoactivatable control of Ca²⁺ signaling and optogenetic
   immunomodulation." *eLife* 4, e10024. <https://elifesciences.org/articles/10024>
   → Opto-CRAC: LOVSoc translocation t½,on = 6.8 ± 2.3 s, t½,off = 28.7 ± 6.5 s; cytosolic Ca²⁺ rise
   t½ 23–36 s, decay t½ ≈ 25–35 s, amplitude ≈ 500–800 nM.

Background, consulted but not used for a number:

10. **Linse S, Helmersson A, Forsén S (1991).** "Calcium binding to calmodulin and its globular
    domains." *J Biol Chem* 266(13), 8050–8054. PMID 1902469.
    → free CaM: C-lobe ~6× higher Ca²⁺ affinity than the N-lobe; cooperativity within each lobe.
11. **Bayley PM, Findlay WA, Martin SR (1996).** "Target recognition by calmodulin: dissecting the
    kinetics and affinity of interaction using short peptide sequences." *Protein Sci* 5(7), 1215–1228.
    PMID 8819155. <https://pmc.ncbi.nlm.nih.gov/articles/PMC2143466>
    → half-site peptides of M13: K_d ≈ 1 µM (N-terminal half) vs ≈ 100 µM (C-terminal half); Ca₂-CaM
    intermediate.
12. **Martin SR, Masino L, Bayley PM (2000).** "Enhancement by Mg²⁺ of domain specificity in
    Ca²⁺-dependent interactions of calmodulin with target sequences." *Protein Sci* 9(12), 2477–2488.
    <https://pmc.ncbi.nlm.nih.gov/articles/PMC2144519>
    → apo-CaM binds skMLCK and neuromodulin sequences with K_d ≈ 70–90 nM at low ionic strength,
    essentially unchanged by 5 mM Mg²⁺.
13. **Peersen OB, Madsen TS, Falke JJ (1997).** "Intermolecular tuning of calmodulin by target peptides
    and proteins: differential effects on Ca²⁺ binding and implications for kinase activation."
    *Protein Sci* 6(4), 794–807.
    → different targets tune the Ca²⁺ affinity of CaM by different factors.
14. **Brown SE, Martin SR, Bayley PM (1997).** "Kinetic control of the dissociation pathway of
    calmodulin-peptide complexes." *J Biol Chem* 272(6), 3389–3397. PMID 9013581.
    → dissociation pathway of CaM–peptide complexes depends on the peptide/CaM ratio.

**Not verified:** the full texts of sources 2, 3, 10, 13 and 14 were not read; their values here come
from the abstracts (Europe PMC). Sources 1, 4–9, 11 and 12 were read in full text or in the parts
containing the numbers quoted.

---

## 7. What each parameter does

Here are the parameters from the report, grouped by what they are.

### Model parameters (`Params` in `qtft/params.py`)

**Calcium protocol — the input signal over time**

- **`ca_rest_uM`** (0.1 µM) — the calcium level during the baseline phase, before the influx starts, which sets how much binding happens "at rest".
- **`ca_peak_uM`** (1.0 µM) — the calcium plateau reached during the influx, which sets the maximum binding strength in the run.
- **`ca_rise_s`** (5 s) — how fast calcium climbs to that plateau after the influx starts (time to 99 %), so it controls how sharply assembly switches on.
- **`ca_tau_s`** (15–20 s) — the decay time constant after the influx ends, which sets how fast the cell pumps calcium back out and thus how fast disassembly begins.

**Calmodulin loading — converting calcium into a switch**

- **`ca_K_uM`** (0.5 µM) — the calcium concentration at which calmodulin is half loaded, i.e. the midpoint of the on/off switch.
- **`ca_hill`** (3) — the Hill coefficient, which sets how *steep* that switch is: higher means a sharper transition from off to on over a narrower calcium range.
- **`f(t)`** (derived, not set) — the resulting fraction of loaded calmodulin, f = Caⁿ/(Kⁿ + Caⁿ), which multiplies into every rate below; it is the single number through which calcium acts in the whole model.

**Bond kinetics — how particles stick together**

- **`k_on`** (1.86·10⁻⁶ ns⁻¹ at f = 1) — the rate at which a Qt and an Ft *already within binding range* form a bond at full calcium loading, so it sets how quickly agglomerates grow.
- **`k_off_ca_per_s`** (10⁻³ s⁻¹) — the rate at which a bond breaks while calmodulin is calcium-loaded, so it sets how stable agglomerates are during the influx.
- **`k_off_free_per_s`** (1.0 s⁻¹) — the rate at which a bond breaks once calcium is gone, so it sets how fast agglomerates fall apart during the pump-out.
- **`R_bind`** (40.5 nm) — the centre-to-centre distance within which a Qt–Ft pair is eligible to bond, i.e. the capture radius of the contact reaction.
- **`maxbond_qt` / `maxbond_ft`** (60 / 10) — the maximum number of bonds one Qt or one Ft can hold, which encodes how many binding sites each cage surface carries and therefore the avidity of a cage–cage contact.

### Literature quantities (what the papers actually measured)

**Affinity**

- **K_d (dissociation constant)** — the concentration at which half the peptide is bound; it is the equilibrium ratio k_off/k_on, so a small K_d means a tight, long-lived bond (0.011 nM for the Trp peptide, 4.1 nM for its Tyr variant).
- **K_a (association constant)** — simply 1/K_d, the form Tsvetkov et al. report for calcium-free calmodulin (10⁶ M⁻¹ without salt, 10³ M⁻¹ at physiological salt).
- **EC50** — the calcium concentration at which a measured *response* is half-maximal; in Atanasijevic's nanoparticles that response is aggregation itself (1.4 µM), which is the closest published analogue to `ca_K_uM` in this model.

**Peptide kinetics (Török & Trentham)**

- **k₊₁ (association rate constant)** — how fast the peptide and calmodulin find each other in solution (8.8·10⁸ M⁻¹s⁻¹, near the diffusion limit); it is the molecular upper bound behind the model's coarse-grained `k_on`.
- **k₋₁ (dissociation rate constant)** — how fast the first, loose encounter complex falls apart again (5.7 s⁻¹) before it has locked in.
- **k₊₂ / k₋₂ (isomerisation rates)** — the slow conformational "locking" step after the initial encounter (≈ 0.4–1 s⁻¹), which is what converts a weak contact into a tight complex.
- **Overall off-rate k₋₁k₋₂/(k₊₂+k₋₂)** — the net rate at which a fully formed complex releases the peptide (0.0079 s⁻¹, a lifetime of ~2 minutes); this is the direct measured counterpart of `k_off_ca_per_s`.

**Calcium kinetics (Johnson, Persechini)**

- **Ca²⁺ off-rate per lobe** — how fast calcium leaves calmodulin's N-lobe (fast, 405 s⁻¹ free) and C-lobe (slow, 2.4 s⁻¹ free); when a target peptide is bound these drop to 1.8–2.9 and 0.1–0.4 s⁻¹, and the slowest of them is what limits peptide release, so it is the counterpart of `k_off_free_per_s`.
- **Ca²⁺ on-rate per lobe** — how fast calcium binds (1.6·10⁸ M⁻¹s⁻¹ N-lobe, 2.3·10⁶ M⁻¹s⁻¹ C-lobe), which is why the model can treat f as following calcium instantly on the second timescale.
- **Affinity enhancement factor** — how much a bound peptide tightens calmodulin's grip on calcium (9-fold for the C-lobe with RS-20), which is the physical reason `ca_K_uM` sits below the value for free calmodulin.

**Protocol kinetics (He et al., optoCRAC)**

- **t½ (half-life) of the calcium rise and decay** — the time for the cytosolic calcium signal to reach or fall by half after the light goes on or off (23–36 s and 25–35 s); these are the measured counterparts of `ca_rise_s` and `ca_tau_s`, and both say your model's protocol is currently several times too fast.
