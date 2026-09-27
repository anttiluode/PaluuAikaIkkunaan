# The Wake Behind the Wave

## A thinking ledger on rotating activity, Martinotti cells, and who gets to continue

*Working paper for Antti Luode — 27 September 2026. This is a mechanism proposal and an evidence ledger, not a claim that cortical vortices have been identified as thoughts or as Martinotti-cell activity.*

### Abstract

A running neural trajectory can change more than the activity of the cells it visits. It can leave those cells refractory and, through recruited inhibition, change the receptivity of other cells nearby. The geometry is suggestive: pyramidal neurons carry deep-layer activity toward apical tufts in layer 1; Martinotti interneurons can receive pyramidal excitation and send ascending, laterally spreading axons to dendrites in that same superficial layer. Cortical rotating waves show that activity can move through a spatially organized network without a clock specifying each site's phase. I propose a narrower bridge between these observations: **a traveling excitatory trajectory may recruit a delayed, lateral inhibitory footprint that changes where incoming apical context can influence the next trajectory.** Two small repository lines make parts of this idea testable, while leaving the proposed biological bridge open. The decisive question is whether the footprint controls competing self-running sequences better than ordinary local gating or matched generic inhibition, without receiving the answer as an input.

## 1. What I think the image is showing us

Picture two geometries crossing. Pyramidal apical trunks rise from deeper cortex into layer 1, where their tufts can receive long-range feedback. Martinotti axons also rise from deeper layers, then ramify laterally and contact dendrites across nearby columns. Deep pyramidal firing can recruit a Martinotti-mediated inhibitory path back onto pyramidal dendrites. In layer-5 pyramidal cells, apical input can alter gain and burst generation when it interacts with somatic activity. These are distinct, experimentally supported pieces of circuitry [1–4].

Now let excitatory activity move across a cortical map. At each location it may alter its own future by refractoriness. If it also recruits Martinotti cells, it could alter the *apical receptivity of other locations*. The activity front and its inhibitory footprint need not coincide. Recruitment, synaptic delay, spatial spread, and recovery determine which tufts are accessible at a given moment.

This is the image that seems worth keeping:

> A trajectory does not merely visit places in order. It may construct a temporary map of where subsequent trajectories can be changed by context.

The footprint is a **hypothesis**. Martinotti arbors are not a demonstrated rotating sheet, and a laterally branching axon is not by itself a traveling wave. A moving excitatory pattern could recruit delayed local inhibition as it passes; whether real cortical waves do so remains to be measured. A footprint also need not be neatly “behind” a wave: broad axonal reach, asymmetric connectivity, recruitment latency, and competing inputs could put part of it beside or ahead of the front.

## 2. Ledger: observation, construction, conjecture

| Entry | What the source establishes | Boundary that matters here |
| --- | --- | --- |
| Rotating cortical activity | Ye and colleagues report rotating 2–8 Hz activity in awake mouse cortex, often centered on somatosensory maps. Local axonal orientation matches the rotation. The published study reports reduced rotation after bilateral disruption of circular circuitry [5]. | This supports an anatomical contribution to *wave geometry*. It does not identify Martinotti cells as wave carriers or show apical gating by the waves. |
| Vertical and lateral inhibition | Martinotti cells in juvenile rat somatosensory cortex have ascending axons that spread horizontally in layer 1; pyramidal-to-Martinotti-to-pyramidal inhibition and control of dendritic calcium events have been measured [1–3]. Delayed lateral inhibition through these cells has also been studied in mouse and human neocortical preparations [4]. | An anatomical route and a synaptic effect do not demonstrate a propagating Martinotti wake during a cortical vortex. SST is a broader interneuron class than Martinotti cells. |
| Distant co-ripples | In human working-memory recordings, roughly 90 Hz ripples co-occur at distant sites and are associated with increased cross-region co-firing and reinstatement of stimulus-related patterns [6]. | These recordings do not demonstrate that a slow somatosensory wave, layer-1 inhibition, or apical calcium events open the co-ripple windows. A 25 ms overlap is part of the paper's co-ripple criterion, not a proven local tuft coincidence rule. |
| Place and learned order | [PaluuAikaIkkunaan Stage 0.5–0.6](https://github.com/anttiluode/PaluuAikaIkkunaan/blob/a50de046ef27ca708bfa71ce3ae6e7257ef3cccc/README.md) puts a reversible, propagating place ring beside a learned, forward chain. The chain fills missing episode words but interferes with backward recall; the README reports two active candidate words in 91% of backward cycles. | The ring topology, one-sided kick, perfect one-shot binding, and several timing parameters are constructed or tuned. This is an engineering test, not a measured cortical circuit. |
| Local arbitration | On the separate [Stage 0.7 `schema-gate` branch](https://github.com/anttiluode/PaluuAikaIkkunaan/blob/0c5cdbb60c72fe76b8d6b2210f7c8b39e47e3062/README.md), episode drive locally mutes the chain's recurrent content pathways while an unbound place leaves them active. It handles backward recall and gap filling better together than any one swept global chain gain in that toy. | The gate reads present episode drive directly. It is a useful performance reference, not an identified SST mechanism or a test of a delayed lateral wake. As of this snapshot the branch is one commit ahead of `main`. |
| Phase made by propagation | A **reported, uncommitted exploratory probe** in the attached discussion reused the clockless Stage 0.5 ring. Reversal changed measured local phase without changing weights; an amplitude sweep found phase-dependent effects of identical weak pulses. | The circular network and initiation are designed; the pulse amplitude was selected after exploratory sweeping. The report has not established naturally formed spatial phase in a less constrained network. |
| Inhibition recruited by activity | [ReturnToTimeWindow's unmerged wake spike](https://github.com/anttiluode/ReturnToTimeWindow/blob/7ae2249be85b5a277a0c944b291dcaaaf460887c/WAKE_SPIKE.md) fixes early episode and later schema slots and gives the early event a delayed apical-inhibition trace. Its 12-seed panel reports 0 intrusion and complete gap filling, while a decision-time-integral-matched tonic arm suppresses both. | The event schedule, threshold, and near-threshold basal/apical decomposition are constructed. The one-lane assay works equally well with **zero lateral spread**. A later, post-canonical two-lane fork makes adjacency matter, but both lanes and their timing are still prescribed. This is a toy existence proof, not a cortical-wave result. |

These entries should remain separate. In particular, the positive wake result cannot be added to the rotating-wave result as though someone had observed the bridge between them.

## 3. A minimal mechanism, with its weak point visible

Let \(E_i(t)\) be the activity of local excitatory population \(i\). Its own refractory state \(R_i(t)\) gives one form of history-dependent susceptibility. Define a *different* variable, \(M_j(t)\), for inhibition delivered to apical dendrites at population \(j\):

\[
M_j(t)=\sum_i W^{M}_{ij}\int_0^\infty k_M(\tau)\,f\!\left(E_i(t-\delta_{ij}-\tau)\right)\,d\tau.
\]

Here \(W^{M}_{ij}\) describes who can inhibit whom, \(\delta_{ij}\) includes recruitment and transmission delay, and \(k_M\) describes how inhibition decays. This is a proposed causal filter, not a measured fit to Martinotti cells. If \(E\) travels, the filter produces a moving \(M\) without storing a global phase variable.

A deliberately limited reader at the next site is

\[
D_j(t)=\operatorname{ReLU}\!\left(B_j(t)-\theta-R_j(t)\right)
\left[1+\alpha A_j(t)\bigl(1-g(M_j(t))\bigr)\right],
\]

where \(B_j\) is basal drive, \(A_j\) is apical/context drive, and \(g(M)\) increases from zero as inhibition rises. The form keeps apical input modulatory: with no supported basal candidate it does not create an item. Internal trajectory dynamics follow from \(D\); publication can remain a separate operation, as in ReturnToTimeWindow's AIS-like control.

The equation exposes the hard part. **If a wrong schema continuation already crosses firing threshold on basal drive alone, tuft inhibition cannot remove it.** Stage 0.6's learned transitions and rebound are basal inputs. Closing apical gain may weaken a competing candidate, yet leave its content active. A successful Stage 0.7 gate actually scales those basal chain pathways. The two interventions must be compared; calling both “SST gating” would erase the difference.

The proposed biological mechanism therefore has a conditional prediction: it can settle the episode/schema conflict through tuft control only where the competing basal path is sufficiently near threshold for apical gain to decide whether it engages. Elsewhere, an additional circuit operation would be required. That possibility is a finding to expose, not something to patch away with a stronger arbitrary inhibition term.

There is also a spatial condition. A delayed field matters to a competing trajectory only when the relevant dendrites lie within its effective connections. The first wake assay's zero-spread success shows that its initial result could be explained by same-site event gating. The second fork makes adjacency consequential *by placing a separate candidate lane next door*. Neither establishes that cortex organizes episode and schema into those lanes.

## 4. Where the physical time enters

There are two potentially independent histories:

1. **Self-history:** the passing population becomes refractory. In a bidirectionally connected ring, its recently active side resists immediate reactivation; this can make an undirected path support a directional wave.
2. **Neighbor-history:** the passing population recruits a delayed inhibitory path to other dendrites. Their receptivity now depends on what happened nearby, including events in cells that never represented the incoming content.

That second history is the more interesting addition. It turns the medium's past into an operator on *future inputs to different cells*. A weak cue delivered twice to the same destination can have different effects because the intervening trajectory changed the local susceptibility field. Reversing or redirecting the wave should change where that cue works, even if the cue, neuron identities, and synaptic weights stay fixed.

The wave paper's approximate 0.05 m/s propagation speed corresponds to about 20 ms per millimeter along the measured wave path in that preparation [5]. This is a useful scale for designing recordings, not an estimated Martinotti delay: the cell types, geometry, and propagation mechanism must be measured independently. A delayed response and a finite spatial arbor do not automatically produce a clean traveling gate.

This is the candidate sense in which **physical time does computation**: a traveling trajectory constructs a fleeting distribution of receptive and unreceptive sites, and later events encounter that distribution. A numerical sequence model could implement the same input-output function. The research question is whether the physical arrangement yields a distinct, reusable causal operation under matched controls.

## 5. One small experiment that could embarrass the idea

The two repos offer a sequence of tests, but the next test needs **one connected machine**. The minimal version would have two self-running, locally coupled population trajectories. An excitatory wave must generate its own visit times. Its activity recruits a delayed lateral \(M\) field continuously. Competing candidates receive basal drive and apical context. No code path may read `episode_present`, a hand-assigned candidate label, a prescribed local phase, or a fixed early/late schema slot to decide whom to suppress.

Keep the Stage 0.6 tasks: backward recall, gaps, noisy continuation, and a competing learned forward route. Measure the internal wave and inhibitory field as well as output, so a gain tweak cannot pass merely by making everything quieter. Compare the following with identical learned content and as closely matched decision-time inhibitory exposure as possible:

| Condition | Question it asks |
| --- | --- |
| Moving, activity-recruited lateral inhibition | Does a trajectory create an aperture for another trajectory? |
| Same-site inhibition | Does spatial reach contribute anything beyond event-triggered local gating? |
| Time-shifted or location-shuffled inhibition | Do the actual delay and neighborhood matter? |
| Tonic or globally timed inhibition | Is activity-contingent space-time structure necessary? |
| Episode-drive gate from Stage 0.7 | Does the new physical mechanism approximate a simpler, already successful local arbitration rule? |
| Basal-only schema test | Does tuft control have enough leverage in this regime to settle the contest at all? |

The primary joint outcome is low **schema intrusion where a trustworthy episode is present** and high **schema fill where it is absent**, while preserving the wave's continuity. Add wrong and uncertain episode bindings: a mechanism that faithfully protects a false memory without any way to recover should not be described as robust memory arbitration. Separate inhibition acting on apical dendrites from inhibition that simply suppresses somatic spikes or kills the recurrent wave.

Freeze the parameter range and thresholds before a new set of seeds. If the Stage 0.7 gate succeeds but the endogenous field does not, we have learned that local arbitration is useful and this proposed physical implementation is insufficient. If tonic or same-site inhibition matches the field, the moving lateral architecture has not earned an explanatory role. If the effect requires a hand-timed schema event, the experiment has not shown that a self-running trajectory can manufacture the address.

## 6. The biological prediction

For an actual cortical mechanism, one would need simultaneous measures of the wave, local pyramidal activity, identified Martinotti/SST activity, and apical dendritic events. The predicted order is **local excitatory activity → recruited inhibition with a measurable lag → reduced or redistributed apical calcium/context effect at spatially connected targets**. Direction reversal or a change of wave path should move the susceptibility pattern accordingly. A timed Martinotti perturbation should change that pattern while permitting at least some underlying excitatory propagation to continue. Those would test the *bridge*, rather than merely the existence of waves and inhibition separately.

The species and circuit boundary matters. The rotating-wave evidence comes from mouse cortical maps. The co-ripple evidence comes from human corticolimbic recordings during working memory. Nested slow-to-fast control is an intriguing later question, but neither paper tests the pathway between these phenomena. Co-ripples should stay downstream of the first falsifier.

## 7. What would the result mean?

If the connected machine survives these tests, I would describe the result carefully: **a traveling computation creates a temporary spatial control surface that changes the causal effect of subsequent inputs**. The novelty claimed for the project would lie in the measured combination and its controls, not in inventing lateral inhibition, excitable waves, or precision-weighted memory.

This gives your inverse-model line a possible physical interface. A desired consequence could ignite a candidate trajectory; local context could make certain continuations more susceptible; a moving inhibitory field could veto or release alternatives; and an output gate could determine which internal result becomes an emitted event. That is a path from *candidate internal sequences* to *controlled sequence expression*. The current tests do not demonstrate an inverse model or neural mirroring. They tell us what machinery those larger ideas would need.

For now, the strongest sentence is also the most modest:

> A moving wave can be investigated as both an activity pattern and a maker of temporary conditions for the next pattern. The cortical mechanism joining those two roles is still an open experiment.

## References and source record

1. Wang Y et al. (2004), [*Anatomical, physiological and molecular properties of Martinotti cells in the somatosensory cortex of the juvenile rat*](https://pubmed.ncbi.nlm.nih.gov/15331670/), *Journal of Physiology*, DOI [10.1113/jphysiol.2004.073353](https://doi.org/10.1113/jphysiol.2004.073353).
2. Silberberg G and Markram H (2007), [*Disynaptic inhibition between neocortical pyramidal cells mediated by Martinotti cells*](https://pubmed.ncbi.nlm.nih.gov/17329212/), *Neuron*.
3. Murayama M et al. (2009), [*Dendritic encoding of sensory stimuli controlled by deep cortical interneurons*](https://pubmed.ncbi.nlm.nih.gov/19151696/), *Nature*. See also Larkum ME et al. (1999), [*A new cellular mechanism for coupling inputs arriving at different cortical layers*](https://pubmed.ncbi.nlm.nih.gov/10192334/), *Nature*.
4. Obermayer J et al. (2018), [*Lateral inhibition by Martinotti interneurons is facilitated by cholinergic inputs in human and mouse neocortex*](https://www.nature.com/articles/s41467-018-06628-w), *Nature Communications*.
5. Ye Z et al. (2026), [*Brain-wide topographic coordination of rotating waves*](https://doi.org/10.1126/science.adx1369), *Science*. The attached *Brain-wide topographic coordination of traveling spiral waves*, bioRxiv v3 (2025), is the earlier version; the published paper adds the reported circuit disruption result.
6. Verzhbinsky IA et al. (2026), [*Cross-region neuron co-firing mediated by ripple oscillations supports distributed working memory representations*](https://doi.org/10.1038/s41593-026-02403-z), *Nature Neuroscience*.
7. Project record: [PaluuAikaIkkunaan main at Stage 0.6](https://github.com/anttiluode/PaluuAikaIkkunaan/tree/a50de046ef27ca708bfa71ce3ae6e7257ef3cccc), [PaluuAikaIkkunaan Stage 0.7 branch snapshot](https://github.com/anttiluode/PaluuAikaIkkunaan/tree/0c5cdbb60c72fe76b8d6b2210f7c8b39e47e3062), [ReturnToTimeWindow G6 and wake-spike branch snapshot](https://github.com/anttiluode/ReturnToTimeWindow/tree/7ae2249be85b5a277a0c944b291dcaaaf460887c). The reported phase probe was supplied in the attached conversation and was explicitly left uncommitted.
