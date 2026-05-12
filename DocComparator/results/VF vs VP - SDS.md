# Technical research papers in computer vision and graphics comparison report
#### Powered by Proxy-Pointer

**Generated:** 2026-05-11<br>
**Criteria:** Score Distillation Sampling (SDS) and optimization mechanics vs. Stroke Style Extraction strategies<br>
**Document 1:** VectorFusion<br>
**Document 2:** VectorPainter<br>
**Document Type:** Technical research papers in computer vision and graphics

---

## Executive Summary

<table style="max-width: 600px; border-collapse: collapse; border: 1px solid #334155;">
  <thead>
    <tr style="background-color: #f8fafc;">
      <th style="border: 1px solid #cbd5e1; padding: 12px; text-align: left;">Metric</th>
      <th style="border: 1px solid #cbd5e1; padding: 12px; text-align: left;">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">VectorFusion Sections Analyzed</td>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">6</td>
    </tr>
    <tr>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">Comparisons Performed</td>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">16</td>
    </tr>
    <tr>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">🔴 Significant Discrepancies</td>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">3</td>
    </tr>
    <tr>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">🟡 Moderate Differences</td>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">13</td>
    </tr>
    <tr>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">🟢 Aligned Sections</td>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">0</td>
    </tr>
  </tbody>
</table>

---

## Detailed Comparison

### Comparison #1 | 3.3. Score distillation sampling

> **VectorFusion excerpt:** 3.3. Score distillation sampling Diffusion models can be trained on arbitrary signals, but it is easier to train them in a space where data is abundant. Standard diffusion samplers like (2) operate in the same space that the diffusion model was trained. While samplers can be modified to solve many...

**Matching VectorPainter sections:** B. Stylized SVG Synthesis, A. Stroke Style Extraction, VI. TOTAL LOSS FUNCTION

**Criteria:** *Score Distillation Sampling (SDS) and optimization mechanics vs. Stroke Style Extraction strategies*

#### ↳ B. Stylized SVG Synthesis

> **VectorPainter excerpt:** B. Stylized SVG Synthesis In this step, we adopt an optimization-based pipeline for SVG synthesis, following the prior work [6]–[8]. Initially, an SVG is created in the vector space and subsequently rendered to the pixel space using DiffVG [14]. In the pixel space, losses are computed, then gradients are backpropagated...

**Discrepancy:** 🔴 SIGNIFICANT DISCREPANCY

**Role:** Optimization Loss Formulation and Prior Integration

**Shared Concepts:** 🤝 *Both frameworks utilize an optimization-based pipeline that leverages differentiable rasterization (DiffVG) to bridge the gap between pixel-space objectives and vector-space parameterization. They share a fundamental reliance on backpropagating gradients from a pretrained image-space prior—specifically diffusion-based models—to refine Bézier primitives.*

**Key Difference:** 📐 *Stochastic Generative Prior (SDS) vs. Deterministic Structural Prior (Stroke Extraction)*

**Analysis:** Because VectorFusion employs Score Distillation Sampling (SDS), it treats the pretrained diffusion model as a frozen teacher that provides a stochastic gradient direction, effectively bypassing the need for a reference image by optimizing toward a text-conditioned distribution. In contrast, VectorPainter shifts the optimization burden from the loss function to the initialization phase; by extracting and preserving explicit style strokes from a reference image, it anchors the optimization in a deterministic structural prior. While VectorFusion trades structural stability for the generative flexibility of the SDS loss, VectorPainter utilizes a dual-constraint strategy—combining Sinkhorn-based optimal transport for stroke-level fidelity with DDIM inversion for global style—to mitigate the "structure disruption" often observed when applying standard style transfer to vector graphics. Ultimately, VectorFusion optimizes for "likelihood under a generative model," whereas VectorPainter optimizes for "alignment with a specific stylistic exemplar."

#### ↳ A. Stroke Style Extraction

> **VectorPainter excerpt:** A. Stroke Style Extraction - 1) ![](figures/fileoutpart11.png) i j i 3 j=1 i i j 3 j=1 P= {p}= {(x, y)}, Stroke Extraction: We define the desired vector graphic S as a collection of n vector strokes S = {Ei}in=1. Specifically, each stroke Ei = {si, ci, wi} is represented...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Optimization Loss Formulation vs. Structural Initialization and Reconstruction

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable vector primitives—specifically Bézier curves—and gradient-based optimization to align a rasterized output with a target visual objective. They share a fundamental reliance on differentiable rendering to bridge the gap between discrete vector parameters and pixel-space loss functions.*

**Key Difference:** 📐 *Stochastic Generative Prior-driven Optimization vs. Deterministic Heuristic-driven Initialization*

**Analysis:** Because VectorFusion utilizes Score Distillation Sampling (SDS), it can synthesize vector content from abstract text prompts by distilling a generative diffusion prior, whereas VectorPainter requires a concrete reference image to drive its superpixel-based stroke extraction. While VectorFusion treats the optimization as a stochastic sampling process where the loss is derived from a pre-trained teacher model, VectorPainter frames the initial stage as a deterministic reconstruction task using a standard MSE loss to match a style reference. Consequently, VectorFusion trades structural predictability for generative flexibility, while VectorPainter ensures high structural alignment by anchoring the optimization to a geometric initialization derived from SLIC segmentation.

#### ↳ VI. TOTAL LOSS FUNCTION

> **VectorPainter excerpt:** VI. TOTAL LOSS FUNCTION Optimal Transport Loss Lot. Unlike pixel-based losses, such as the ℓ2 loss that computes the mean squared error (MSE) for each corresponding pixel, the minimum transportation loss provides a more effective measure of similarity between the canvas and the reference image. ![](figures/fileoutpart22.png) n×n + n x...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Optimization Loss Formulation and Guidance Strategy

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rendering to optimize the parameters of vector primitives through gradient-based refinement. They share a common objective of aligning a parametric image representation with a target visual distribution by backpropagating losses defined in the pixel space—whether derived from a generative prior or a reference image—to the underlying stroke parameters.*

**Key Difference:** 📐 *Stochastic Generative Distillation vs. Deterministic Structural Alignment*

**Analysis:** Because VectorFusion employs Score Distillation Sampling (SDS), it bypasses the need for a concrete reference image by distilling the gradients of a pretrained diffusion model directly into the vector parameters, effectively treating the diffusion model as a probabilistic teacher. In contrast, VectorPainter utilizes a multi-term loss function comprising Optimal Transport (Sinkhorn distance) and $\ell_2$ pixel loss to enforce structural and color fidelity relative to a specific reference or DDIM-inverted sample. While VectorFusion focuses on the zero-shot synthesis of vector graphics from text prompts via stochastic gradient estimation, VectorPainter prioritizes the "labor cost" of stroke placement, trading the generative flexibility of SDS for the precise geometric alignment offered by optimal transport mechanics. Consequently, VectorFusion is better suited for creative synthesis where no target image exists, whereas VectorPainter provides a more robust framework for style transfer and stroke-based reconstruction where structural preservation is paramount.

---


### Comparison #2 | 4.2. Sampling vector graphics by optimization

> **VectorFusion excerpt:** 4.2. Sampling vector graphics by optimization The pipeline in 4.1 is flawed since samples may not be easily representable by a set of paths. Figure 4 illustrates the problem. Conditioned on text, a diffusion model produces samples from the distribution pφ(x|y). Vectorization with LIVE finds a SVG with a close...

**Matching VectorPainter sections:** B. Stylized SVG Synthesis, A. Stroke Style Extraction, VIII. STROKE STYLE EXTRACTION ALGORITHM

**Criteria:** *Score Distillation Sampling (SDS) and optimization mechanics vs. Stroke Style Extraction strategies*

#### ↳ B. Stylized SVG Synthesis

> **VectorPainter excerpt:** B. Stylized SVG Synthesis In this step, we adopt an optimization-based pipeline for SVG synthesis, following the prior work [6]–[8]. Initially, an SVG is created in the vector space and subsequently rendered to the pixel space using DiffVG [14]. In the pixel space, losses are computed, then gradients are backpropagated...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Optimization Framework and Loss Formulation for SVG Synthesis

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rasterization (DiffVG) to bridge the gap between discrete vector primitives and pixel-based loss functions, and both rely on Latent Diffusion Models (LDM) to provide high-level semantic or stylistic guidance during the optimization of Bézier path parameters.*

**Key Difference:** 📐 *Stochastic Score Distillation vs. Deterministic Style-Preserving Initialization*

**Analysis:** While VectorFusion employs Score Distillation Sampling (SDS) to stochastically distill a text-conditioned diffusion prior into vector paths, VectorPainter shifts the focus toward style fidelity by initializing the optimization with pre-extracted style strokes and regularizing the process via Sinkhorn-based optimal transport. Because VectorFusion prioritizes the alignment of the SVG with a text prompt through latent-space gradient descent, it accepts the risk of information loss during vectorization, whereas VectorPainter mitigates this by using a reference-based style-preserving loss to maintain structural and stylistic consistency. Ultimately, VectorFusion trades specific stylistic control for generative flexibility, while VectorPainter leverages a more constrained initialization to ensure the output mirrors the intricate details of a reference image.

#### ↳ A. Stroke Style Extraction

> **VectorPainter excerpt:** A. Stroke Style Extraction - 1) ![](figures/fileoutpart11.png) i j i 3 j=1 i i j 3 j=1 P= {p}= {(x, y)}, Stroke Extraction: We define the desired vector graphic S as a collection of n vector strokes S = {Ei}in=1. Specifically, each stroke Ei = {si, ci, wi} is represented...

**Discrepancy:** 🔴 SIGNIFICANT DISCREPANCY

**Role:** Optimization and Initialization Frameworks for Vector Representation

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable vector primitives (Bézier paths) and gradient-based optimization to refine the parameters of a vector graphic. They share a common objective of bridging the gap between raster-based visual information and scalable vector representations, employing iterative refinement loops to minimize a defined loss function—whether that loss is derived from a generative diffusion prior or a reconstruction error against a reference image.*

**Key Difference:** 📐 *Generative Distillation (Stochastic/Prior-driven) vs. Structural Reconstruction (Deterministic/Reference-driven)*

**Analysis:** Because VectorFusion relies on a pre-trained latent diffusion prior to synthesize graphics from text, it necessitates a complex Score Distillation Sampling (SDS) mechanism that backpropagates through a frozen LDM encoder and a differentiable renderer. While VectorFusion optimizes for semantic alignment with a caption, VectorPainter shifts the focus toward structural imitation, utilizing superpixel segmentation (SLIC) to provide a deterministic initialization for stroke extraction. VectorFusion trades the structural precision of a reference image for generative flexibility, whereas VectorPainter prioritizes high-fidelity reconstruction by optimizing a mean-square-error loss between the rasterized strokes and the original style reference.

#### ↳ VIII. STROKE STYLE EXTRACTION ALGORITHM

> **VectorPainter excerpt:** VIII. STROKE STYLE EXTRACTION ALGORITHM We summarize the Stroke Style Extraction Algorithm in Alg. S1. The algorithm consists of two main steps: stroke extraction (lines 1–11) and stroke vectorization (lines 12–16). During the stroke extraction step, strokes are extracted from each segmented region after super-pixel segmentation, and the control points,...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Generative Optimization Mechanics vs. Structural Extraction Pipeline

**Shared Concepts:** 🤝 *Both frameworks aim to parameterize vector primitives—specifically Bézier paths and stroke attributes—to represent visual content in a scalable SVG format. They share the high-level objective of bridging the gap between pixel-based representations (latent distributions or reference images) and structured vector outputs.*

**Key Difference:** 📐 *Stochastic Gradient-based Refinement vs. Deterministic Heuristic Extraction*

**Analysis:** Because VectorFusion utilizes Score Distillation Sampling (SDS) to optimize path parameters, it requires a differentiable rendering pipeline (DiffVG) and backpropagation through a frozen Latent Diffusion Model (LDM) encoder to align the vector output with a text prompt. While VectorFusion treats vector synthesis as a generative sampling problem guided by a diffusion prior, VectorPainter shifts the methodology toward a two-stage extraction process that relies on super-pixel segmentation and attribute imitation to capture stroke-level style. VectorFusion trades the computational speed of direct extraction for the semantic flexibility of a generative prior, whereas VectorPainter prioritizes structural fidelity by explicitly decomposing a reference image into control points and color attributes. Ultimately, VectorFusion optimizes a global loss function in latent space, while VectorPainter employs a more localized, bottom-up strategy to reconstruct style from segmented regions.

---


### Comparison #3 | 4.4. Stylizing by constraining vector representation

> **VectorFusion excerpt:** 4.4. Stylizing by constraining vector representation Users can control the style of art generated by VectorFu-sion by modifying the input text, or by constraining the set of primitives and parameters that can be optimized. The choice of SVG vector primitives determines the level of abstraction of the result. We explore...

**Matching VectorPainter sections:** III. METHODOLOGY

**Criteria:** *Score Distillation Sampling (SDS) and optimization mechanics vs. Stroke Style Extraction strategies*

#### ↳ III. METHODOLOGY

> **VectorPainter excerpt:** III. METHODOLOGY In this section, we introduce VectorPainter for stylized vector graphics synthesis. Given a text prompt P and a reference painting image Is, VectorPainter aims to generate a vector graphic S whose content aligns with the text prompt while the style remains consistent with the reference image. Vec-torPainter comprises...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Stylization Strategy and Optimization Framework

**Shared Concepts:** 🤝 *Both frameworks leverage differentiable rasterization to optimize Bézier primitives (paths or strokes) through gradient-based refinement, aiming to produce stylized vector graphics that align with semantic prompts.*

**Key Difference:** 📐 *Constraint-based Abstraction (VectorFusion) vs. Exemplar-based Imitation (VectorPainter)*

**Analysis:** While VectorFusion achieves stylization by restricting the primitive parameter space—such as path count and stroke width—to force abstract representations like iconography or sketches, VectorPainter shifts the burden of style to an explicit extraction phase using SLIC superpixels and imitation learning. Because VectorFusion relies on the generative prior of Score Distillation Sampling (SDS) to fill in stylistic gaps within its constrained geometry, it offers higher flexibility for text-only workflows; conversely, VectorPainter trades this generative freedom for high-fidelity style consistency by anchoring the optimization to a reference image via Sinkhorn distance and DDIM inversion. Ultimately, VectorFusion treats style as a byproduct of geometric constraints, whereas VectorPainter treats it as a transferable signal extracted from a source image.

---


### Comparison #4 | F.3. Optimization

> **VectorFusion excerpt:** F.3. Optimization We optimize with a batch size of 1, allowing VectorFusion to run on a single low-end GPU with at least 10 GB of memory. VectorFusion uses the Adam optimizer with β1 = 0.9, β2 = 0.9,  = 10−6 . On an NVIDIA RTX 2080ti GPU, VectorFusion (SD...

**Matching VectorPainter sections:** B. Stylized SVG Synthesis, A. Stroke Style Extraction, VII. IMPLEMENTATION DETAILS

**Criteria:** *Score Distillation Sampling (SDS) and optimization mechanics vs. Stroke Style Extraction strategies*

#### ↳ B. Stylized SVG Synthesis

> **VectorPainter excerpt:** B. Stylized SVG Synthesis In this step, we adopt an optimization-based pipeline for SVG synthesis, following the prior work [6]–[8]. Initially, an SVG is created in the vector space and subsequently rendered to the pixel space using DiffVG [14]. In the pixel space, losses are computed, then gradients are backpropagated...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Optimization Strategy and Loss Formulation

**Shared Concepts:** 🤝 *Both frameworks leverage differentiable rasterization (DiffVG) to optimize SVG primitives via gradient-based refinement in pixel space. They share a common objective of synthesizing high-quality vector graphics by backpropagating losses from a rendered image back to stroke parameters, specifically targeting the optimization of control point coordinates and color values.*

**Key Difference:** 📐 *Stochastic Score Distillation (SDS) vs. Deterministic Style-Preserving Priors*

**Analysis:** Because VectorFusion relies on Score Distillation Sampling (SDS) to hallucinate content from text prompts, it necessitates a highly tuned learning rate schedule and saturation penalties to stabilize the stochastic gradients inherent in diffusion-based distillation. While VectorFusion treats the optimization as a generative task starting from varied initializations, VectorPainter shifts the focus toward style-conditioned reconstruction, utilizing explicit style-stroke initialization and Sinkhorn-based optimal transport to constrain the optimization manifold. VectorPainter trades the generative flexibility of SDS for structural fidelity by anchoring the optimization to a reference image's style priors, whereas VectorFusion prioritizes structural fluidity through higher coordinate learning rates to allow the SDS loss to reshape the SVG geometry.

#### ↳ A. Stroke Style Extraction

> **VectorPainter excerpt:** A. Stroke Style Extraction - 1) ![](figures/fileoutpart11.png) i j i 3 j=1 i i j 3 j=1 P= {p}= {(x, y)}, Stroke Extraction: We define the desired vector graphic S as a collection of n vector strokes S = {Ei}in=1. Specifically, each stroke Ei = {si, ci, wi} is represented...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Optimization Mechanics vs. Structural Initialization

**Shared Concepts:** 🤝 *Both frameworks utilize Bézier primitives and gradient-based refinement to achieve structural fidelity in vector graphics. They share a reliance on differentiable rendering to bridge the gap between discrete vector parameters and pixel-based loss functions—specifically, VectorFusion optimizes via Score Distillation Sampling (SDS) while VectorPainter employs a mean-square-error (MSE) "imitation learning" phase for initial reconstruction.*

**Key Difference:** 📐 *Stochastic Optimization vs. Deterministic Initialization*

**Analysis:** Because VectorFusion relies on SDS to distill priors from a diffusion model, it necessitates a complex learning rate schedule and specific saturation penalties to manage the high-variance gradients inherent in the distillation process. While VectorFusion treats the vector canvas as a space for global refinement via text-conditioning, VectorPainter shifts the focus to a data-driven initialization where superpixel segmentation (SLIC) dictates the initial topology and color distribution. VectorPainter trades the flexibility of pure text-to-SVG generation for higher structural consistency with a reference image by employing an "imitation learning" phase, whereas VectorFusion prioritizes the optimization dynamics—such as cosine decay and warmup—to ensure the SDS loss converges to a coherent aesthetic without manual stroke extraction.

#### ↳ VII. IMPLEMENTATION DETAILS

> **VectorPainter excerpt:** VII. IMPLEMENTATION DETAILS Our method accepts a textual prompt to express semantics and a reference image to control the style. It is based on an optimization-based vector graphics synthesis pipeline [7] with a differentiable rasterizer R [14], and style transfer methods in pixel space, InstantStyle [16] and StyleAligned [17]. As...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Optimization Configuration and Implementation Details

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rasterization to optimize vector primitives (Bézier curves/strokes) against a pre-trained diffusion model prior. They share a common objective of refining control points and color parameters through gradient-based optimization to achieve semantic alignment with a text prompt.*

**Key Difference:** 📐 *Parameter Scheduling vs. Style-Conditioned Initialization*

**Analysis:** While VectorFusion employs a sophisticated learning rate schedule—utilizing linear warmup and cosine decay to facilitate structural exploration followed by fine-grained refinement—VectorPainter relies on a static learning rate coupled with a high stroke count (up to 3,000) to capture complex painterly textures. Because VectorFusion is designed for iconography and sketches, it prioritizes structural flexibility through coordinate-heavy learning rates, whereas VectorPainter shifts the focus toward style fidelity by integrating DDIM inversion and style-preserving losses. VectorPainter trades the computational efficiency of VectorFusion’s low-memory SDS approach for the high-fidelity stylistic control offered by SDXL and specialized style-transfer modules.

---


### Comparison #5 | 5.5. Sketches and line drawings

> **VectorFusion excerpt:** 5.5. Sketches and line drawings Figure 2 includes line drawing samples. VectorFusion produces recognizable and clear sketches from scratch without any image reference, even complex scenes with multiple objects. In addition, it is able to ignore distractor terms irrel- evant to sketches, such as “watercolor” or “Brightly colored” and capture...

**Matching VectorPainter sections:** C. Qualitative Evaluation, B. Comparison Baselines, A. Stroke Style Extraction

**Criteria:** *Score Distillation Sampling (SDS) and optimization mechanics vs. Stroke Style Extraction strategies*

#### ↳ C. Qualitative Evaluation

> **VectorPainter excerpt:** C. Qualitative Evaluation From Fig. 5, we can make the following observations: (1) The results of StyleCLIPDraw are poor, indicating the challenges of directly synthesizing stylized vector graphics. (2) Most raster-image-oriented style transfer methods preserve the reference’s style well. However, their results are in raster format. Once vectorization is performed,...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Qualitative Experimental Evaluation

**Shared Concepts:** 🤝 *Both frameworks leverage text-guided optimization to synthesize vector primitives, focusing on the semantic alignment between a textual prompt and the resulting SVG geometry. They share a reliance on differentiable rasterization to bridge the gap between discrete vector paths and the pixel-based loss functions derived from large-scale vision-language models.*

**Key Difference:** 📐 *Text-only semantic distillation vs. Reference-conditioned style extraction.*

**Analysis:** Because VectorFusion prioritizes zero-shot synthesis from text, it relies on the latent prior of the diffusion model to filter out stylistic distractors and isolate structural line work. While VectorFusion demonstrates robustness in generating sketches without external visual cues, VectorPainter argues that text prompts alone are insufficient for precise stylistic control, instead proposing a dual-conditioning scheme that incorporates a reference image. VectorPainter trades the simplicity of a text-only interface for higher stylistic fidelity, whereas VectorFusion optimizes for semantic purity by ignoring irrelevant stylistic keywords in the prompt to maintain sketch-like sparsity.

#### ↳ B. Comparison Baselines

> **VectorPainter excerpt:** B. Comparison Baselines To synthesize stylized vector graphics, there are mainly three approaches: (1) Synthesis though Text Prompt and Reference Image. Like the existing method StyleCLIPDraw [3] and our VectorPainter, these methods generate stylized vector graphics directly based on a given text and a reference image. (2) Ras-terization then Vectorization....

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Experimental Methodology and Baseline Comparison

**Shared Concepts:** 🤝 *Both frameworks operate within the paradigm of optimization-based vector graphics synthesis, leveraging differentiable rendering to align vector primitives with high-level semantic or stylistic objectives.*

**Key Difference:** 📐 *Conditioning Disparity (Zero-shot Semantic Distillation vs. Reference-based Style Supervision)*

**Analysis:** Because VectorFusion leverages the inherent semantic priors of diffusion models via Score Distillation Sampling, it achieves stylistic outputs like sketches through text-only prompts without requiring external visual guidance. In contrast, VectorPainter adopts a multi-modal approach, framing the problem as a style transfer task that necessitates explicit reference images and specialized loss functions to supervise the optimization of vector primitives. This creates a fundamental tradeoff: VectorFusion offers greater generative autonomy at the cost of precise style control, while VectorPainter ensures stylistic fidelity by anchoring the optimization to a concrete visual reference.

#### ↳ A. Stroke Style Extraction

> **VectorPainter excerpt:** A. Stroke Style Extraction - 1) ![](figures/fileoutpart11.png) i j i 3 j=1 i i j 3 j=1 P= {p}= {(x, y)}, Stroke Extraction: We define the desired vector graphic S as a collection of n vector strokes S = {Ei}in=1. Specifically, each stroke Ei = {si, ci, wi} is represented...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Methodology vs. Experimental Validation of Stylized Vector Generation

**Shared Concepts:** 🤝 *Both frameworks utilize Bézier primitives as the fundamental unit for stylized vector output and rely on the optimization of these primitives to achieve visual coherence. They share a common objective of translating high-level stylistic intent—whether derived from text or a reference image—into a sparse set of vector strokes.*

**Key Difference:** 📐 *Zero-shot Semantic Distillation vs. Reference-based Structural Reconstruction*

**Analysis:** Because VectorFusion leverages the generative prior of a diffusion model via SDS, it can synthesize abstract sketches from purely semantic text prompts, effectively ignoring distractor terms to maintain conceptual clarity. In contrast, VectorPainter adopts a deterministic initialization strategy, employing SLIC superpixel segmentation to extract stroke topology directly from a reference image before refining it through imitation learning. While VectorFusion trades structural precision for the flexibility of zero-shot generation, VectorPainter prioritizes stylistic fidelity and spatial alignment by anchoring its optimization to the low-level pixel attributes of a source image.

---


### Comparison #6 | C. Ablation: Number of paths

> **VectorFusion excerpt:** C. Ablation: Number of paths VectorFusion optimizes path coordinates and colors, but the number of primitive paths is a non-differentiable hyper-parameter. Vector graphics with fewer paths will be more abstract, whereas photorealism and details can be improved with many paths. In this ablation, we experiment with different number of paths....

**Matching VectorPainter sections:** E. Ablation Studies, B. Stylized SVG Synthesis, A. Stroke Style Extraction

**Criteria:** *Score Distillation Sampling (SDS) and optimization mechanics vs. Stroke Style Extraction strategies*

#### ↳ E. Ablation Studies

> **VectorPainter excerpt:** E. Ablation Studies 1) Effect of Imitation Learning Strategy. Our imitation learning strategy aims to ensure that strokes extracted from the reference image authentically capture the desired style. As shown in Fig. 7(a), without this strategy, the extracted strokes inadequately reflect the reference style, resulting in noticeable blank holes in...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Ablation Analysis of Initialization Strategies and Representation Capacity.

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rendering to optimize Bézier-based primitives and employ ablation studies to validate their respective initialization and optimization heuristics. They share a common objective of improving the visual coherence and fidelity of vector outputs, whether conditioned on text (VectorFusion) or reference images (VectorPainter), and both identify random initialization as a baseline that leads to suboptimal coverage or disorganized geometry.*

**Key Difference:** 📐 *Representation-density optimization (VectorFusion) vs. Structural-prior initialization (VectorPainter).*

**Analysis:** Because VectorFusion treats the number of paths as a discrete, non-differentiable hyper-parameter, it relies on path reinitialization and scheduling to overcome local minima in the Score Distillation Sampling (SDS) landscape. While VectorFusion evaluates the trade-off between abstraction and caption consistency through primitive density, VectorPainter shifts the focus toward structural fidelity by leveraging imitation learning and Optimal Transport to anchor strokes to a reference style. VectorFusion trades structural constraints for semantic flexibility via SDS, whereas VectorPainter prioritizes style-preserving priors and DDIM inversion to prevent the "disorganized strokes" and "blank holes" that occur when the optimization lacks a strong geometric starting point.

#### ↳ B. Stylized SVG Synthesis

> **VectorPainter excerpt:** B. Stylized SVG Synthesis In this step, we adopt an optimization-based pipeline for SVG synthesis, following the prior work [6]–[8]. Initially, an SVG is created in the vector space and subsequently rendered to the pixel space using DiffVG [14]. In the pixel space, losses are computed, then gradients are backpropagated...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Optimization Strategy and Initialization Heuristics.

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rasterization to bridge the gap between vector primitives (Bézier paths) and pixel-space objectives, employing gradient-based optimization to refine stroke parameters. They both recognize initialization as a critical bottleneck for convergence and quality, addressing it through either tracing (LIVE) or style-stroke extraction to provide a better starting point than random noise.*

**Key Difference:** 📐 *Semantic-driven abstraction (VectorFusion) vs. Reference-driven style preservation (VectorPainter).*

**Analysis:** While VectorFusion treats the number of paths as a non-differentiable hyperparameter to be ablated for semantic consistency, VectorPainter views the initial stroke set as a carrier of style priors that must be constrained via Sinkhorn distances. Because VectorFusion prioritizes caption alignment through SDS and path reinitialization, it accepts higher geometric variance to maximize primitive usage; conversely, VectorPainter introduces a style-preserving loss to explicitly penalize deviations from the reference stroke topology. VectorFusion trades structural rigidity for semantic fidelity by dynamically reinitializing paths, whereas VectorPainter trades optimization flexibility for stylistic continuity by anchoring the strokes to a reference image's latent and geometric distribution.

#### ↳ A. Stroke Style Extraction

> **VectorPainter excerpt:** A. Stroke Style Extraction - 1) ![](figures/fileoutpart11.png) i j i 3 j=1 i i j 3 j=1 P= {p}= {(x, y)}, Stroke Extraction: We define the desired vector graphic S as a collection of n vector strokes S = {Ei}in=1. Specifically, each stroke Ei = {si, ci, wi} is represented...

**Discrepancy:** 🔴 SIGNIFICANT DISCREPANCY

**Role:** Primitive Initialization and Optimization Strategy

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable vector primitives (specifically Bézier curves) and gradient-based optimization to bridge the gap between vector representations and rasterized outputs. They share a fundamental reliance on determining the optimal number and placement of paths to achieve visual fidelity, whether guided by text-based Score Distillation Sampling (SDS) or image-based imitation learning.*

**Key Difference:** 📐 *Stochastic Optimization-heavy (VectorFusion) vs. Deterministic Segmentation-heavy (VectorPainter)*

**Analysis:** Because VectorFusion operates in a text-to-SVG generative context, it treats the number of paths as a non-differentiable hyperparameter and employs a reinitialization heuristic to maximize primitive utility during the SDS process. While VectorFusion focuses on the trade-off between abstraction and caption consistency through path count ablations, VectorPainter shifts the complexity to the initialization phase by using SLIC superpixel segmentation to deterministically extract stroke topology from a reference image. VectorPainter trades the generative flexibility of SDS for structural alignment via "Vectorized Stroke Imitation Learning," whereas VectorFusion accepts the inherent abstraction of its primitives to maintain semantic coherence with a text prompt.

---

