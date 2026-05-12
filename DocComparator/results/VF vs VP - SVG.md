# Research paper comparison report
#### Powered by Proxy-Pointer

**Generated:** 2026-05-12<br>
**Criteria:** Compare how each paper approaches style control and primitive initialization in vector graphics synthesis. Specifically, analyze how VectorFusion uses path reinitialization and raster sample initialization versus how VectorPainter extracts and rearranges vectorized strokes from a reference image using stroke imitation learning and style-preserving losses<br>
**Document 1:** VectorFusion<br>
**Document 2:** VectorPainter<br>
**Document Type:** research paper

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
      <td style="border: 1px solid #cbd5e1; padding: 12px;">10</td>
    </tr>
    <tr>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">Comparisons Performed</td>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">30</td>
    </tr>
    <tr>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">🔴 Significant Discrepancies</td>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">7</td>
    </tr>
    <tr>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">🟡 Moderate Differences</td>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">23</td>
    </tr>
    <tr>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">🟢 Aligned Sections</td>
      <td style="border: 1px solid #cbd5e1; padding: 12px;">0</td>
    </tr>
  </tbody>
</table>

---

## Detailed Comparison

### Comparison #1 | 4.3. Reinitializing paths

> **VectorFusion excerpt:** 4.3. Reinitializing paths In our most flexible setting, synthesizing flat iconographic vectors, we allow path control points, fill colors and SVG background color to be optimized. During the course of optimization, many paths learn low opacity or shrink to a small area and are unused. To encourage usage of paths...

**Matching VectorPainter sections:** B. Stylized SVG Synthesis, A. Stroke Style Extraction, VI. TOTAL LOSS FUNCTION

**Criteria:** *Compare how each paper approaches style control and primitive initialization in vector graphics synthesis. Specifically, analyze how VectorFusion uses path reinitialization and raster sample initialization versus how VectorPainter extracts and rearranges vectorized strokes from a reference image using stroke imitation learning and style-preserving losses*

#### ↳ B. Stylized SVG Synthesis

> **VectorPainter excerpt:** B. Stylized SVG Synthesis In this step, we adopt an optimization-based pipeline for SVG synthesis, following the prior work [6]–[8]. Initially, an SVG is created in the vector space and subsequently rendered to the pixel space using DiffVG [14]. In the pixel space, losses are computed, then gradients are backpropagated...

**Discrepancy:** 🔴 SIGNIFICANT DISCREPANCY

**Role:** Primitive Initialization and Optimization Heuristics

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rasterization (DiffVG) within an optimization-based pipeline to refine Bézier primitives via backpropagated gradients. They share the objective of managing primitive behavior—specifically path placement and utility—to ensure the final SVG accurately represents the target aesthetic while avoiding degenerate solutions like unused or invisible paths.*

**Key Difference:** 📐 *Stochastic Pruning (VectorFusion) vs. Structural Prior-Driven Initialization (VectorPainter)*

**Analysis:** VectorFusion addresses the "dead path" problem in flat iconography by employing a stochastic reinitialization heuristic, where paths falling below opacity or area thresholds are pruned and randomly redistributed to encourage canvas exploration. Conversely, VectorPainter mitigates the optimization challenge by replacing random initialization with a deterministic style prior, extracting vectorized strokes directly from a reference image to provide a high-fidelity structural starting point. Because VectorFusion lacks a reference-image constraint, it relies on periodic resets to maintain detail; meanwhile, VectorPainter introduces a dual-level style-preserving loss—utilizing Sinkhorn distance for stroke-level topology and DDIM inversion for global consistency—to ensure the optimization trajectory does not deviate from the source style. Ultimately, VectorFusion trades structural precision for generative flexibility in text-to-SVG tasks, whereas VectorPainter prioritizes stylistic mimesis by anchoring the optimization within a pre-defined stroke manifold.

#### ↳ A. Stroke Style Extraction

> **VectorPainter excerpt:** A. Stroke Style Extraction - 1) ![](figures/fileoutpart11.png) i j i 3 j=1 i i j 3 j=1 P= {p}= {(x, y)}, Stroke Extraction: We define the desired vector graphic S as a collection of n vector strokes S = {Ei}in=1. Specifically, each stroke Ei = {si, ci, wi} is represented...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Optimization Heuristics

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable vector primitives (Bézier curves/paths) and gradient-based optimization to synthesize vector graphics. They both address the fundamental challenge of primitive placement and utility—VectorFusion through dynamic re-seeding of underutilized paths and VectorPainter through structured extraction of strokes from a reference image.*

**Key Difference:** 📐 *Stochastic Re-seeding vs. Deterministic Style Extraction*

**Analysis:** While VectorFusion employs a stochastic reinitialization heuristic to prevent path collapse and ensure canvas coverage during open-ended optimization, VectorPainter adopts a deterministic, reference-driven approach by using SLIC superpixels to extract and vectorize strokes that mimic a specific style. Because VectorFusion prioritizes path utility through the random re-insertion of "dead" paths (those with low opacity or area), it maintains high visual complexity without requiring a structural prior; conversely, VectorPainter trades this flexibility for structural fidelity by grounding its initial primitive state in the geometry and color distribution of a style reference image. Ultimately, VectorFusion treats initialization as a dynamic maintenance task to avoid local minima, whereas VectorPainter treats it as a supervised imitation learning problem to ensure style consistency.

#### ↳ VI. TOTAL LOSS FUNCTION

> **VectorPainter excerpt:** VI. TOTAL LOSS FUNCTION Optimal Transport Loss Lot. Unlike pixel-based losses, such as the ℓ2 loss that computes the mean squared error (MSE) for each corresponding pixel, the minimum transportation loss provides a more effective measure of similarity between the canvas and the reference image. ![](figures/fileoutpart22.png) n×n + n x...

**Discrepancy:** 🔴 SIGNIFICANT DISCREPANCY

**Role:** Primitive Management and Optimization Objective

**Shared Concepts:** 🤝 *Both frameworks leverage differentiable rendering to optimize vector primitives (paths or strokes) within a gradient-based refinement loop. They share the objective of ensuring high primitive utility—preventing "dead" or redundant paths—to achieve visual complexity and fidelity.*

**Key Difference:** 📐 *Stochastic Heuristic Reinitialization vs. Distribution-Matching Loss*

**Analysis:** Because VectorFusion often encounters path collapse or "vanishing" gradients during Score Distillation Sampling, it employs a discrete, threshold-based reinitialization heuristic that stochastically replaces underutilized paths with random circles to maintain image density. While VectorFusion relies on this periodic "reset" to escape local minima, VectorPainter adopts a more continuous approach by framing stroke arrangement as an Optimal Transport problem, using the Sinkhorn distance to minimize the "labor cost" of moving strokes into alignment with a reference image. VectorFusion trades off structural continuity for increased detail through its random re-injection mechanism, whereas VectorPainter prioritizes stylistic preservation by integrating the transportation loss directly into the total objective function alongside pixel-level $L_2$ constraints. Consequently, VectorFusion’s initialization is a dynamic maintenance task during optimization, while VectorPainter’s approach is a structured distribution-matching strategy designed to mimic the stroke characteristics of a source style.

---


### Comparison #2 | B. Ablation: Reinitializing paths

> **VectorFusion excerpt:** B. Ablation: Reinitializing paths We reinitialize paths below an opacity threshold or area threshold periodically, every 50 iterations. The purpose of reinitializing small, faint paths is to encourage the usage of all paths. Path are not reinitialized for the final 200-500 iterations of optimization, so reinitialized paths have enough time...

**Matching VectorPainter sections:** E. Ablation Studies, B. Stylized SVG Synthesis, A. Stroke Style Extraction

**Criteria:** *Compare how each paper approaches style control and primitive initialization in vector graphics synthesis. Specifically, analyze how VectorFusion uses path reinitialization and raster sample initialization versus how VectorPainter extracts and rearranges vectorized strokes from a reference image using stroke imitation learning and style-preserving losses*

#### ↳ E. Ablation Studies

> **VectorPainter excerpt:** E. Ablation Studies 1) Effect of Imitation Learning Strategy. Our imitation learning strategy aims to ensure that strokes extracted from the reference image authentically capture the desired style. As shown in Fig. 7(a), without this strategy, the extracted strokes inadequately reflect the reference style, resulting in noticeable blank holes in...

**Discrepancy:** 🔴 SIGNIFICANT DISCREPANCY

**Role:** Primitive Initialization and Optimization Heuristics

**Shared Concepts:** 🤝 *Both frameworks utilize gradient-based optimization of Bézier primitives to synthesize vector graphics and employ ablation studies to validate that specialized initialization strategies outperform random primitive placement. They share the objective of ensuring "path coverage"—preventing empty regions or underutilized primitives—through the management of stroke properties during the optimization pipeline.*

**Key Difference:** 📐 *Dynamic Stochastic Re-sampling (VectorFusion) vs. Static Reference-based Seeding (VectorPainter)*

**Analysis:** Because VectorFusion primarily operates in a text-to-SVG generative context using Score Distillation Sampling (SDS), it treats primitive management as a dynamic maintenance task, periodically reinitializing faint or small paths to prevent local minima and ensure full canvas utilization. In contrast, VectorPainter adopts a style-transfer paradigm where initialization is a deterministic extraction process; it leverages stroke imitation learning and Optimal Transport loss to anchor the optimization to the topological structure of a reference image. While VectorFusion trades initial stylistic precision for generative flexibility by using raster-traced samples (LIVE) or random paths as a baseline, VectorPainter prioritizes stylistic continuity by enforcing a structural prior through DDIM inversion and style-preserving losses, ensuring that the synthesized strokes inherit the specific aesthetic DNA of the source material.

#### ↳ B. Stylized SVG Synthesis

> **VectorPainter excerpt:** B. Stylized SVG Synthesis In this step, we adopt an optimization-based pipeline for SVG synthesis, following the prior work [6]–[8]. Initially, an SVG is created in the vector space and subsequently rendered to the pixel space using DiffVG [14]. In the pixel space, losses are computed, then gradients are backpropagated...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Optimization Heuristics

**Shared Concepts:** 🤝 *Both frameworks leverage differentiable rasterization (DiffVG) to bridge the gap between vector primitives and pixel-based loss functions, employing gradient-based optimization to refine Bézier parameters. They share a common objective of overcoming the limitations of random initialization—VectorFusion through dynamic path recycling and VectorPainter through reference-based stroke priors.*

**Key Difference:** 📐 *Stochastic Path Recycling vs. Deterministic Style-Prior Initialization*

**Analysis:** Because VectorFusion relies on Score Distillation Sampling (SDS) which can lead to "dead" or redundant paths, it implements a periodic reinitialization heuristic based on opacity and area thresholds to maximize path utility and improve R-Precision. While VectorFusion treats initialization as a dynamic process of pruning and respawning to ensure convergence, VectorPainter shifts the burden to the pre-optimization phase by extracting vectorized strokes directly from a reference image via stroke imitation learning. VectorPainter trades the flexibility of "from-scratch" generation for high-fidelity style preservation using Sinkhorn-based optimal transport losses to constrain stroke movement, whereas VectorFusion maintains a more generalizable pipeline that uses raster samples (LIVE) primarily as a structural starting point rather than a rigid stylistic constraint.

#### ↳ A. Stroke Style Extraction

> **VectorPainter excerpt:** A. Stroke Style Extraction - 1) ![](figures/fileoutpart11.png) i j i 3 j=1 i i j 3 j=1 P= {p}= {(x, y)}, Stroke Extraction: We define the desired vector graphic S as a collection of n vector strokes S = {Ei}in=1. Specifically, each stroke Ei = {si, ci, wi} is represented...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Optimization Heuristics

**Shared Concepts:** 🤝 *Both frameworks leverage differentiable rasterization to optimize Bézier primitives and employ specific initialization strategies to overcome the non-convexity of the vector optimization landscape. They share a common goal of ensuring that the available primitive budget is utilized effectively, using iterative refinement to align the vector output with a target visual or conceptual distribution.*

**Key Difference:** 📐 *Heuristic-driven pruning (VectorFusion) vs. Data-driven structural extraction (VectorPainter).*

**Analysis:** Because VectorFusion primarily operates within a text-to-SVG paradigm using Score Distillation Sampling (SDS), it encounters the "dead path" problem where primitives become transparent or negligible; it addresses this through a stochastic reinitialization heuristic that resets paths failing to meet opacity or area thresholds every 50 iterations. While VectorFusion treats initialization as a dynamic maintenance task to ensure primitive utility during synthesis, VectorPainter adopts a deterministic, reference-heavy approach by using SLIC superpixel segmentation to extract the initial stroke topology directly from a style image. VectorPainter trades the generative flexibility of VectorFusion’s diffusion-based priors for structural fidelity, employing "imitation learning" to minimize MSE between the rasterized strokes and the reference, whereas VectorFusion relies on periodic re-seeding to escape local minima in the SDS loss landscape. Ultimately, VectorFusion optimizes for conceptual coverage through path recycling, while VectorPainter optimizes for stylistic mimicry through precise geometric extraction.

---


### Comparison #3 | F.1. Path initialization

> **VectorFusion excerpt:** F.1. Path initialization Iconographic Art We initialize our closed B´ezier paths with radius 20, random fill color, and opacity uniformly sampled between 0.7 and 1. Paths have 4 segments. Pixel Art Pixel art is represented with a 32×32 grid of square polygons. The coordinates of square vertices are not optimized....

**Matching VectorPainter sections:** VII. IMPLEMENTATION DETAILS, B. Stylized SVG Synthesis, A. Stroke Style Extraction

**Criteria:** *Compare how each paper approaches style control and primitive initialization in vector graphics synthesis. Specifically, analyze how VectorFusion uses path reinitialization and raster sample initialization versus how VectorPainter extracts and rearranges vectorized strokes from a reference image using stroke imitation learning and style-preserving losses*

#### ↳ VII. IMPLEMENTATION DETAILS

> **VectorPainter excerpt:** VII. IMPLEMENTATION DETAILS Our method accepts a textual prompt to express semantics and a reference image to control the style. It is based on an optimization-based vector graphics synthesis pipeline [7] with a differentiable rasterizer R [14], and style transfer methods in pixel space, InstantStyle [16] and StyleAligned [17]. As...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Style Conditioning Strategy

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rasterization within an optimization-based pipeline to refine Bézier primitives (strokes or paths) according to target visual objectives. They share a fundamental reliance on gradient-based updates to control point coordinates and color properties, leveraging pre-trained diffusion models to guide the synthesis of vector graphics.*

**Key Difference:** 📐 *Stochastic Geometric Priors vs. Reference-Driven Imitation Learning*

**Analysis:** While VectorFusion relies on stochastic geometric priors—such as randomly placed circles or fixed grids—to seed the optimization process, VectorPainter adopts a more structured approach by first vectorizing a reference image through a dedicated imitation learning phase. Because VectorFusion constrains style through primitive-specific hyperparameters, such as fixed stroke widths for sketches or grid-aligned polygons for pixel art, it offers a more automated but less stylistically flexible pipeline compared to VectorPainter, which trades computational overhead for high-fidelity style transfer by extracting and rearranging thousands of strokes from a visual exemplar. While VectorFusion optimizes from a "blank canvas" of random paths guided by text, VectorPainter shifts the burden of style definition from manual primitive constraints to a reference-conditioned optimization loss, ensuring that the final vector output inherits the specific brushwork and texture of the source image.

#### ↳ B. Stylized SVG Synthesis

> **VectorPainter excerpt:** B. Stylized SVG Synthesis In this step, we adopt an optimization-based pipeline for SVG synthesis, following the prior work [6]–[8]. Initially, an SVG is created in the vector space and subsequently rendered to the pixel space using DiffVG [14]. In the pixel space, losses are computed, then gradients are backpropagated...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Stylistic Constraint Formulation

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rasterization to bridge the gap between vector primitive parameters and pixel-level objectives, optimizing Bézier-based paths to achieve specific aesthetic outcomes. They share a fundamental reliance on initialization as a mechanism to reduce the optimization search space and guide the solver toward a desired stylistic manifold.*

**Key Difference:** 📐 *Heuristic-based geometric priors (VectorFusion) vs. Reference-driven data priors (VectorPainter).*

**Analysis:** VectorFusion enforces style through rigid, category-specific geometric heuristics—such as fixed stroke widths for sketches or grid-constrained polygons for pixel art—effectively treating initialization as a structural boundary for the optimization. Conversely, VectorPainter shifts the stylistic burden from manual heuristics to a data-driven extraction process, where vectorized strokes are harvested from a reference image and maintained via an Optimal Transport (Sinkhorn) loss. Because VectorFusion lacks a reference-image style prior, it relies on stochastic sampling within predefined geometric bounds, whereas VectorPainter’s use of DDIM inversion and style-preserving losses allows it to maintain high-fidelity stylistic imitation that transcends simple category-based constraints.

#### ↳ A. Stroke Style Extraction

> **VectorPainter excerpt:** A. Stroke Style Extraction - 1) ![](figures/fileoutpart11.png) i j i 3 j=1 i i j 3 j=1 P= {p}= {(x, y)}, Stroke Extraction: We define the desired vector graphic S as a collection of n vector strokes S = {Ei}in=1. Specifically, each stroke Ei = {si, ci, wi} is represented...

**Discrepancy:** 🔴 SIGNIFICANT DISCREPANCY

**Role:** Primitive Initialization and Style Conditioning

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rasterization to optimize Bézier primitives—specifically control points, color, and opacity—within a gradient-based refinement loop. They share the objective of defining a starting state for vector primitives that constrains the optimization toward a specific artistic style (e.g., sketches, iconography, or painterly strokes).*

**Key Difference:** 📐 *Stochastic Heuristic vs. Reference-Driven Determinism*

**Analysis:** Because VectorFusion is designed for text-to-SVG synthesis via Score Distillation Sampling, it employs a stochastic initialization strategy where primitives are placed according to generic geometric priors, such as random fill colors for icons or fixed-width black curves for sketches. While VectorFusion relies on the diffusion prior to evolve these "blank slate" primitives into a coherent style, VectorPainter shifts the stylistic burden to a reference image, using SLIC superpixel segmentation to extract and vectorize strokes that inherit the color and thickness of the source material. This methodological divergence creates a significant tradeoff: VectorFusion offers greater generative flexibility by starting from uninformative grids or random paths, whereas VectorPainter ensures superior structural alignment and style consistency by using a "vectorized stroke imitation learning" phase to pre-optimize primitives against a target raster image before the final synthesis.

---


### Comparison #4 | 4.4. Stylizing by constraining vector representation

> **VectorFusion excerpt:** 4.4. Stylizing by constraining vector representation Users can control the style of art generated by VectorFu-sion by modifying the input text, or by constraining the set of primitives and parameters that can be optimized. The choice of SVG vector primitives determines the level of abstraction of the result. We explore...

**Matching VectorPainter sections:** B. Stylized SVG Synthesis, A. Stroke Style Extraction, VI. TOTAL LOSS FUNCTION

**Criteria:** *Compare how each paper approaches style control and primitive initialization in vector graphics synthesis. Specifically, analyze how VectorFusion uses path reinitialization and raster sample initialization versus how VectorPainter extracts and rearranges vectorized strokes from a reference image using stroke imitation learning and style-preserving losses*

#### ↳ B. Stylized SVG Synthesis

> **VectorPainter excerpt:** B. Stylized SVG Synthesis In this step, we adopt an optimization-based pipeline for SVG synthesis, following the prior work [6]–[8]. Initially, an SVG is created in the vector space and subsequently rendered to the pixel space using DiffVG [14]. In the pixel space, losses are computed, then gradients are backpropagated...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Stylistic Constraint Methodology

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rasterization (DiffVG) to bridge the gap between vector primitives and pixel-based loss functions, optimizing Bézier parameters through gradient-based refinement. They share a common objective of achieving stylistic abstraction by manipulating the number, type, and constraints of vector primitives within an optimization-based pipeline.*

**Key Difference:** 📐 *Constraint-driven Abstraction (VectorFusion) vs. Reference-driven Imitation (VectorPainter)*

**Analysis:** Because VectorFusion relies on text-to-SVG synthesis via Score Distillation Sampling (SDS), it achieves style control primarily through architectural constraints—such as limiting path counts for iconography or fixing stroke widths for sketches—and employs iterative reinitialization via the LIVE autovectorization phase to ensure semantic alignment. While VectorFusion treats style as an emergent property of these primitive constraints and textual prompts, VectorPainter formalizes style as a transferable prior, extracting specific vectorized strokes from a reference image to initialize the optimization. VectorPainter trades the zero-shot generative flexibility of VectorFusion for high-fidelity style preservation, utilizing a dual-level loss strategy: a stroke-level Sinkhorn distance to discourage excessive geometric variation and a global-level DDIM inversion to maintain latent style consistency. Ultimately, VectorFusion optimizes for "style-by-limitation" to reach abstract representations, whereas VectorPainter optimizes for "style-by-imitation" to ensure the synthesized vector graphics mirror the intricate details of a source aesthetic.

#### ↳ A. Stroke Style Extraction

> **VectorPainter excerpt:** A. Stroke Style Extraction - 1) ![](figures/fileoutpart11.png) i j i 3 j=1 i i j 3 j=1 P= {p}= {(x, y)}, Stroke Extraction: We define the desired vector graphic S as a collection of n vector strokes S = {Ei}in=1. Specifically, each stroke Ei = {si, ci, wi} is represented...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Style Control Strategy

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rasterization to optimize Bézier primitives (paths or strokes) and employ gradient-based refinement to align vector outputs with a target visual representation. They both leverage the inherent constraints of vector primitives—such as path count, width, and color—as a mechanism for abstraction and style control.*

**Key Difference:** 📐 *Generative Constraint-based Optimization (VectorFusion) vs. Reconstructive Decomposition-based Initialization (VectorPainter)*

**Analysis:** While VectorFusion achieves style control by imposing architectural constraints on primitives (e.g., limiting path counts or fixing stroke widths) and initializing via diffusion-based rejection sampling, VectorPainter adopts a data-driven initialization strategy by decomposing a reference image into superpixels to extract latent stroke geometries. Because VectorFusion relies on Score Distillation Sampling (SDS), its style is emergent from the interaction between text prompts and primitive limitations; conversely, VectorPainter ensures style consistency through "stroke imitation learning," where the initial vector state is a direct geometric approximation of a source image's texture and color distribution. This represents a tradeoff between the creative flexibility of VectorFusion’s prior-driven synthesis and the high-fidelity style transfer enabled by VectorPainter’s structural decomposition.

#### ↳ VI. TOTAL LOSS FUNCTION

> **VectorPainter excerpt:** VI. TOTAL LOSS FUNCTION Optimal Transport Loss Lot. Unlike pixel-based losses, such as the ℓ2 loss that computes the mean squared error (MSE) for each corresponding pixel, the minimum transportation loss provides a more effective measure of similarity between the canvas and the reference image. ![](figures/fileoutpart22.png) n×n + n x...

**Discrepancy:** 🔴 SIGNIFICANT DISCREPANCY

**Role:** Optimization Objective and Style Control Strategy

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rendering to optimize vector primitives (such as Bézier paths or strokes) through gradient-based refinement. They share the objective of achieving stylistic abstraction by manipulating the parameters of these primitives—such as coordinates and color—to align with a target visual representation.*

**Key Difference:** 📐 *Generative Prior (SDS) vs. Reference-Based Rearrangement (Optimal Transport)*

**Analysis:** VectorFusion achieves style control primarily through structural constraints on the vector primitives themselves, such as limiting path counts for iconography or enforcing grid-based squares for pixel art. In contrast, VectorPainter enforces style by minimizing the Sinkhorn distance between a rendered canvas and a reference image, treating style preservation as an optimal transport problem of moving stroke masses. Because VectorFusion relies on a text-conditioned diffusion prior (SDS), it can initialize primitives from scratch or via iterative rejection sampling to generate novel imagery; conversely, VectorPainter’s use of a style-preserving loss ($L_{sp}$) prioritizes the spatial and color distribution of a specific reference, trading the generative flexibility of VectorFusion for high-fidelity imitation of a source style.

---


### Comparison #5 | 4.1. A baseline: text-to-image-to-vector

> **VectorFusion excerpt:** 4.1. A baseline: text-to-image-to-vector We start by developing a two stage pipeline: sampling an image from Stable Diffusion, then vectorizing it automatically. Given text, we sample a raster image from Stable Diffusion with a Runge-Kutta solver [17] in 50 sampling steps with guidance scale ω = 7.5 (the default settings...

**Matching VectorPainter sections:** B. Stylized SVG Synthesis, A. Stroke Style Extraction, VI. TOTAL LOSS FUNCTION

**Criteria:** *Compare how each paper approaches style control and primitive initialization in vector graphics synthesis. Specifically, analyze how VectorFusion uses path reinitialization and raster sample initialization versus how VectorPainter extracts and rearranges vectorized strokes from a reference image using stroke imitation learning and style-preserving losses*

#### ↳ B. Stylized SVG Synthesis

> **VectorPainter excerpt:** B. Stylized SVG Synthesis In this step, we adopt an optimization-based pipeline for SVG synthesis, following the prior work [6]–[8]. Initially, an SVG is created in the vector space and subsequently rendered to the pixel space using DiffVG [14]. In the pixel space, losses are computed, then gradients are backpropagated...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Style Control Strategy

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rasterization (DiffVG) to optimize Bézier primitives through gradient-based refinement. They share a common objective of bridging the gap between raster representations and vector graphics by initializing paths from a raster source—either generated via diffusion or provided as a reference—and subsequently optimizing their parameters to match a target aesthetic.*

**Key Difference:** 📐 *Generative-Stochastic (VectorFusion) vs. Reference-Deterministic (VectorPainter) initialization.*

**Analysis:** While VectorFusion relies on a text-to-image-to-vector pipeline that uses prompt engineering and staged path reinitialization (LIVE) to handle abstract styles, VectorPainter adopts a more rigid style-transfer paradigm by extracting and preserving specific stroke geometries from a reference image. Because VectorFusion optimizes against a latent score distillation loss, it prioritizes semantic coherence over structural preservation, whereas VectorPainter utilizes Sinkhorn-based optimal transport losses to ensure that the synthesized strokes maintain the topological "DNA" of the source style. Consequently, VectorFusion trades off precise style replication for generative flexibility, while VectorPainter sacrifices generative variety to achieve high-fidelity style consistency through explicit stroke-level constraints.

#### ↳ A. Stroke Style Extraction

> **VectorPainter excerpt:** A. Stroke Style Extraction - 1) ![](figures/fileoutpart11.png) i j i 3 j=1 i i j 3 j=1 P= {p}= {(x, y)}, Stroke Extraction: We define the desired vector graphic S as a collection of n vector strokes S = {Ei}in=1. Specifically, each stroke Ei = {si, ci, wi} is represented...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Style Extraction

**Shared Concepts:** 🤝 *Both frameworks leverage differentiable rendering to optimize Bézier primitives via gradient descent, aiming to minimize the reconstruction error (L2/MSE) between a raster target and a vector representation. They both employ a multi-stage pipeline where an initial raster-like representation is decomposed into discrete vector paths or strokes which are then iteratively refined.*

**Key Difference:** 📐 *Generative-Prior-Driven vs. Exemplar-Decomposition-Driven*

**Analysis:** While VectorFusion initializes primitives by sampling a generative diffusion prior and iteratively adding paths to high-loss regions via the LIVE algorithm, VectorPainter adopts a bottom-up approach by decomposing a static reference image into superpixels to derive initial stroke geometry. Because VectorFusion relies on prompt engineering and latent score distillation to enforce style, it trades the structural fidelity of a specific reference for the creative flexibility of text-to-SVG synthesis. Conversely, VectorPainter’s imitation learning ensures higher stylistic alignment with a source image by explicitly extracting stroke width and color from segmented regions, shifting the optimization burden from "discovering" the image structure to "refining" extracted primitives. VectorFusion's path reinitialization is fundamentally a stochastic search for coverage, whereas VectorPainter's initialization is a deterministic geometric approximation of an existing exemplar.

#### ↳ VI. TOTAL LOSS FUNCTION

> **VectorPainter excerpt:** VI. TOTAL LOSS FUNCTION Optimal Transport Loss Lot. Unlike pixel-based losses, such as the ℓ2 loss that computes the mean squared error (MSE) for each corresponding pixel, the minimum transportation loss provides a more effective measure of similarity between the canvas and the reference image. ![](figures/fileoutpart22.png) n×n + n x...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Optimization Loss Formulation

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rendering to bridge the gap between raster targets and vector primitives, employing gradient-based optimization to refine path parameters. They each rely on a reference raster image—generated via Stable Diffusion in VectorFusion and provided as a reference or DDIM-inverted sample in VectorPainter—to guide the initial geometry and color of the vector elements.*

**Key Difference:** 📐 *Localized Greedy Reconstruction (VectorFusion) vs. Global Distribution Alignment (VectorPainter)*

**Analysis:** Because VectorFusion utilizes the LIVE framework for its baseline initialization, it necessitates a staged, greedy path addition process where new primitives are localized to high-loss regions using a distance-weighted L2 metric. While VectorFusion relies on prompt engineering and Latent Score Distillation to enforce a "flat" vector style post-initialization, VectorPainter internalizes style control within its loss architecture by employing an Optimal Transport (Sinkhorn) distance. This allows VectorPainter to treat strokes as transportable mass, prioritizing the preservation of stylistic "labor" and global distribution over the rigid, pixel-wise reconstruction favored by VectorFusion’s L2-heavy approach. Ultimately, VectorFusion trades global structural fluidity for precise local fidelity, whereas VectorPainter’s style-preserving loss offers a more flexible mechanism for rearranging strokes to match the artistic character of a reference.

---


### Comparison #6 | 2. Related Work

> **VectorFusion excerpt:** 2. Related Work A few works have used pretrained vision-language models to guide vector graphic generation. VectorAscent [11] and CLIPDraw [4] optimize CLIP’s image-text similarity metric [27] to generate vector graphics from text prompts, with a procedure similar to DeepDream [23] and CLIP feature visualization [5]. StyleCLIPDraw [35] extends CLIP-Draw...

**Matching VectorPainter sections:** A. Vector Graphics Synthesis, B. Style Transfer, B. Comparison Baselines

**Criteria:** *Compare how each paper approaches style control and primitive initialization in vector graphics synthesis. Specifically, analyze how VectorFusion uses path reinitialization and raster sample initialization versus how VectorPainter extracts and rearranges vectorized strokes from a reference image using stroke imitation learning and style-preserving losses*

#### ↳ A. Vector Graphics Synthesis

> **VectorPainter excerpt:** A. Vector Graphics Synthesis Scalable Vector Graphics (SVGs) are comprised of essential components such as B´ezier curves, lines, shapes, and colors to represent images. The latest technique for generating SVGs involves using a differentiable rasterizer such as DiffVG [14]. DiffVG bridges the gap between vector graphics and raster image spaces,...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Literature Review and Methodological Contextualization.

**Shared Concepts:** 🤝 *Both frameworks anchor their methodology in differentiable rasterization—specifically the DiffVG engine—to optimize Bézier primitives and leverage pretrained vision-language or diffusion models (CLIP, Stable Diffusion) as the primary supervisory signals for synthesis.*

**Key Difference:** 📐 *Generative Prior Optimization (VectorFusion) vs. Structural Component Supervision (VectorPainter).*

**Analysis:** VectorFusion situates its approach within the evolution of Score Distillation Sampling (SDS) and iterative path optimization derived from LIVE, prioritizing the transition from discriminative to generative priors to overcome the limitations of text-to-image fidelity. Conversely, VectorPainter frames the synthesis process through the fundamental components of SVGs—Bézier curves and lines—emphasizing the differentiable rasterizer's role in bridging the vector-raster divide to facilitate stroke-level control. While VectorFusion’s reliance on diffusion priors necessitates a focus on path reinitialization to escape local minima during optimization, VectorPainter’s emphasis on SVG "essential components" sets the stage for its stroke imitation learning, which extracts stylistic signatures directly from reference images.

#### ↳ B. Style Transfer

> **VectorPainter excerpt:** B. Style Transfer Style Transfer is a task in computer vision that involves combining a content image and a style image to create a new image that preserves the former’s content and the latter’s style patterns. Over the years, many researchers have proposed various models [15]–[18] to improve the quality...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Literature Review and Methodological Positioning

**Shared Concepts:** 🤝 *Both frameworks position themselves within the lineage of differentiable vector synthesis, specifically citing StyleCLIPDraw as a common point of departure for stylized vector generation. They share a fundamental reliance on optimizing Bézier primitives through gradient-based refinement to bridge the gap between raster-based visual priors and resolution-independent vector outputs.*

**Key Difference:** 📐 *Generative Prior (Diffusion-based) vs. Exemplar-based Prior (Style Transfer)*

**Analysis:** While VectorFusion leverages the stochastic latent space of Stable Diffusion to provide a generative prior for path optimization, VectorPainter adopts a more deterministic style-transfer paradigm focused on the explicit extraction of stroke characteristics from reference images. Because VectorFusion extends the LIVE framework, it prioritizes global semantic alignment via Score Distillation Sampling (SDS), effectively using the diffusion model as a high-level critic for path reinitialization. Conversely, VectorPainter trades this broad generative flexibility for granular stylistic fidelity, shifting the focus toward stroke imitation learning and style-preserving losses to ensure that vectorized primitives maintain the textural and morphological integrity of a specific style exemplar. Whereas VectorFusion treats style as an emergent property of the generative prior, VectorPainter treats it as a structural constraint to be extracted and rearranged.

#### ↳ B. Comparison Baselines

> **VectorPainter excerpt:** B. Comparison Baselines To synthesize stylized vector graphics, there are mainly three approaches: (1) Synthesis though Text Prompt and Reference Image. Like the existing method StyleCLIPDraw [3] and our VectorPainter, these methods generate stylized vector graphics directly based on a given text and a reference image. (2) Ras-terization then Vectorization....

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Literature Review and Methodological Contextualization.

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rasterization (DiffVG) as the underlying rendering engine to bridge the gap between discrete vector primitives and gradient-based optimization. They share a common methodological lineage, referencing "Layer-wise Image Vectorization" (LIVE) and "StyleCLIPDraw" as foundational benchmarks for synthesizing Bézier-based graphics through iterative refinement guided by high-level semantic or stylistic objectives.*

**Key Difference:** 📐 *Generative Prior (Diffusion-based) vs. Explicit Style Supervision (Reference-based).*

**Analysis:** VectorFusion shifts the synthesis paradigm toward a generative approach by employing Stable Diffusion as a transferable prior, using Score Distillation Sampling (SDS) to optimize paths without requiring a direct target image. Because VectorFusion relies on the latent knowledge of a diffusion model, it utilizes path reinitialization and raster sample initialization to prevent local minima and ensure the vector primitives converge on the complex textures suggested by the text prompt. While VectorFusion leverages these stochastic generative priors, VectorPainter adopts a more deterministic style-transfer framework, focusing on the extraction and rearrangement of vectorized strokes from a reference image. VectorPainter trades the broad creative flexibility of a diffusion model for high-fidelity aesthetic control, utilizing stroke imitation learning and explicit style-preserving losses (such as STROTSS or NST) to constrain the optimization process to a specific visual reference. Consequently, the initialization in VectorFusion is a mechanism to explore a generative manifold, whereas in VectorPainter, it serves as a structural template for stylistic imitation.

---


### Comparison #7 | 4.2. Sampling vector graphics by optimization

> **VectorFusion excerpt:** 4.2. Sampling vector graphics by optimization The pipeline in 4.1 is flawed since samples may not be easily representable by a set of paths. Figure 4 illustrates the problem. Conditioned on text, a diffusion model produces samples from the distribution pφ(x|y). Vectorization with LIVE finds a SVG with a close...

**Matching VectorPainter sections:** B. Stylized SVG Synthesis, A. Stroke Style Extraction, VI. TOTAL LOSS FUNCTION

**Criteria:** *Compare how each paper approaches style control and primitive initialization in vector graphics synthesis. Specifically, analyze how VectorFusion uses path reinitialization and raster sample initialization versus how VectorPainter extracts and rearranges vectorized strokes from a reference image using stroke imitation learning and style-preserving losses*

#### ↳ B. Stylized SVG Synthesis

> **VectorPainter excerpt:** B. Stylized SVG Synthesis In this step, we adopt an optimization-based pipeline for SVG synthesis, following the prior work [6]–[8]. Initially, an SVG is created in the vector space and subsequently rendered to the pixel space using DiffVG [14]. In the pixel space, losses are computed, then gradients are backpropagated...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Style Control Strategy

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rasterization (DiffVG) to bridge the gap between discrete vector primitives and pixel-based loss functions. They share a reliance on Latent Diffusion Models (LDM) to provide structural or stylistic priors and employ gradient-based optimization to iteratively refine Bézier path parameters.*

**Key Difference:** 📐 *Generative Prior (SDS) vs. Exemplar-Based Prior (Stroke Imitation)*

**Analysis:** Because VectorFusion relies on Score Distillation Sampling (SDS) to extract knowledge from a text-conditioned diffusion model, its style is emergent from the generative prior rather than explicitly constrained by a reference image. While VectorFusion initializes paths to be optimized against a latent-space loss to ensure text-image alignment, VectorPainter adopts a more deterministic initialization by extracting vectorized strokes directly from a reference image to serve as a structural and stylistic anchor. VectorPainter trades the open-ended generative flexibility of VectorFusion for high-fidelity style preservation, utilizing Sinkhorn distances for stroke-level constraints and DDIM inversion for global consistency to ensure the final SVG maintains the topological and aesthetic characteristics of the source strokes.

#### ↳ A. Stroke Style Extraction

> **VectorPainter excerpt:** A. Stroke Style Extraction - 1) ![](figures/fileoutpart11.png) i j i 3 j=1 i i j 3 j=1 P= {p}= {(x, y)}, Stroke Extraction: We define the desired vector graphic S as a collection of n vector strokes S = {Ei}in=1. Specifically, each stroke Ei = {si, ci, wi} is represented...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Style Conditioning Strategy

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rasterization (DiffVG) to bridge the gap between discrete vector primitives and continuous pixel-based loss functions. They share a common optimization backbone where Bézier curve parameters—including control points, color, and stroke width—are refined via gradient descent to minimize a discrepancy metric between the rendered output and a target representation.*

**Key Difference:** 📐 *Stochastic Generative Prior (VectorFusion) vs. Deterministic Structural Extraction (VectorPainter)*

**Analysis:** VectorFusion adopts a top-down generative approach, leveraging Score Distillation Sampling (SDS) to distill semantic knowledge from a latent diffusion model into vector paths, which necessitates backpropagating through a frozen LDM encoder to update the SVG parameters. Conversely, VectorPainter employs a bottom-up reconstruction strategy, using SLIC superpixel segmentation to decompose a reference image into discrete regions that serve as the direct geometric basis for stroke initialization. Because VectorFusion relies on a text-conditioned diffusion prior, it prioritizes semantic flexibility and "hallucinated" detail, whereas VectorPainter’s use of stroke imitation learning and MSE-based reconstruction trades this generative breadth for high-fidelity structural alignment with a specific style reference. Ultimately, while VectorFusion optimizes for a distribution defined by a prompt, VectorPainter optimizes for a specific instance defined by the spatial and textural attributes of the extracted superpixels.

#### ↳ VI. TOTAL LOSS FUNCTION

> **VectorPainter excerpt:** VI. TOTAL LOSS FUNCTION Optimal Transport Loss Lot. Unlike pixel-based losses, such as the ℓ2 loss that computes the mean squared error (MSE) for each corresponding pixel, the minimum transportation loss provides a more effective measure of similarity between the canvas and the reference image. ![](figures/fileoutpart22.png) n×n + n x...

**Discrepancy:** 🔴 SIGNIFICANT DISCREPANCY

**Role:** Optimization Loss Formulation and Primitive Refinement

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rendering to bridge the gap between discrete vector primitives and raster-based loss functions, employing gradient-based optimization to refine path parameters ($\theta$). They each leverage diffusion-based priors—VectorFusion via Score Distillation Sampling (SDS) and VectorPainter via DDIM inversion samples—to guide the synthesis process toward high-quality visual outputs.*

**Key Difference:** 📐 *Stochastic distribution matching (VectorFusion) vs. Deterministic structural alignment (VectorPainter).*

**Analysis:** Because VectorFusion optimizes SVG paths through the lens of a latent-space SDS loss, it prioritizes semantic coherence with text prompts by distilling gradients from a teacher diffusion model directly into the path parameters. While VectorFusion treats the vectorizer as a generative agent guided by probability densities, VectorPainter shifts the focus to structural imitation, employing a Sinkhorn-based Optimal Transport loss to minimize the "labor cost" of rearranging strokes to match a reference image. This creates a fundamental tradeoff: VectorFusion offers superior creative flexibility for text-to-SVG synthesis at the cost of potential structural instability, whereas VectorPainter ensures high-fidelity style preservation by explicitly penalizing spatial and pixel-level deviations from a target reference.

---


### Comparison #8 | 3.1. Vector representation and rendering pipeline

> **VectorFusion excerpt:** 3.1. Vector representation and rendering pipeline Vector graphics are composed of primitives. For our work, we use paths of segments delineated by control points. We configure the control point positions, shape fill color, stroke width and stroke color. Most of our experiments use closed Bezier´ curves. Different artistic styles are...

**Matching VectorPainter sections:** B. Stylized SVG Synthesis, A. Stroke Style Extraction, A. Vector Graphics Synthesis

**Criteria:** *Compare how each paper approaches style control and primitive initialization in vector graphics synthesis. Specifically, analyze how VectorFusion uses path reinitialization and raster sample initialization versus how VectorPainter extracts and rearranges vectorized strokes from a reference image using stroke imitation learning and style-preserving losses*

#### ↳ B. Stylized SVG Synthesis

> **VectorPainter excerpt:** B. Stylized SVG Synthesis In this step, we adopt an optimization-based pipeline for SVG synthesis, following the prior work [6]–[8]. Initially, an SVG is created in the vector space and subsequently rendered to the pixel space using DiffVG [14]. In the pixel space, losses are computed, then gradients are backpropagated...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** SVG Synthesis Methodology and Initialization Strategy

**Shared Concepts:** 🤝 *Both frameworks utilize DiffVG as a differentiable rasterization engine to bridge the gap between vector primitives and pixel-based loss functions. They share a common optimization-based pipeline where gradients are backpropagated from a rendered image to the parameters of Bézier paths (control points, stroke width, and color) to achieve stylized synthesis.*

**Key Difference:** 📐 *Reference-anchored initialization vs. Stochastic generative initialization*

**Analysis:** While VectorFusion achieves stylistic diversity by selecting specific primitive types (e.g., squares for pixel art or unclosed curves for line art) and evolving them via generative priors, VectorPainter adopts a more deterministic approach by initializing the optimization with vectorized strokes extracted directly from a reference image. Because VectorPainter incorporates a style-preserving loss—specifically utilizing the Sinkhorn distance for stroke-level constraints—it maintains structural fidelity to the reference style that VectorFusion’s path reinitialization might otherwise discard in favor of text-prompt alignment. VectorFusion trades the strict structural preservation of a reference for the flexibility of text-to-SVG synthesis, whereas VectorPainter optimizes for "stroke imitation," ensuring that the synthesized output inherits the specific brushwork and texture of the source material through global DDIM inversion and local optimal transport losses.

#### ↳ A. Stroke Style Extraction

> **VectorPainter excerpt:** A. Stroke Style Extraction - 1) ![](figures/fileoutpart11.png) i j i 3 j=1 i i j 3 j=1 P= {p}= {(x, y)}, Stroke Extraction: We define the desired vector graphic S as a collection of n vector strokes S = {Ei}in=1. Specifically, each stroke Ei = {si, ci, wi} is represented...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Style Representation

**Shared Concepts:** 🤝 *Both frameworks utilize Bézier curves as the fundamental geometric primitive and rely on differentiable rasterization to bridge the gap between vector parameters and pixel-based loss functions. They share a common parameterization scheme involving control points, stroke width, and color optimization to represent complex visual information through a sparse set of primitives.*

**Key Difference:** 📐 *Deterministic Reference-Based Extraction (VectorPainter) vs. Stochastic/Sampling-Based Initialization (VectorFusion).*

**Analysis:** While VectorFusion adopts a flexible approach by selecting primitive types (e.g., closed curves vs. line art) to define style and relies on path reinitialization to escape local minima during optimization, VectorPainter implements a deterministic "Stroke Imitation Learning" phase that extracts structural priors directly from a reference image via SLIC superpixel segmentation. Because VectorPainter prioritizes style consistency through the reconstruction of a specific reference, it trades the generative breadth of VectorFusion’s sampling-based initialization for high-fidelity structural alignment. Consequently, VectorFusion’s pipeline is designed for diverse synthesis from abstract prompts, whereas VectorPainter’s architecture is optimized for the precise transfer of existing stroke characteristics into the vector domain.

#### ↳ A. Vector Graphics Synthesis

> **VectorPainter excerpt:** A. Vector Graphics Synthesis Scalable Vector Graphics (SVGs) are comprised of essential components such as B´ezier curves, lines, shapes, and colors to represent images. The latest technique for generating SVGs involves using a differentiable rasterizer such as DiffVG [14]. DiffVG bridges the gap between vector graphics and raster image spaces,...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Technical Foundation and Primitive Definition

**Shared Concepts:** 🤝 *Both frameworks adopt DiffVG as the core differentiable rendering engine to bridge the vector-raster gap and utilize Bézier curves as the fundamental geometric primitive for synthesis.*

**Key Difference:** 📐 *Geometric Constraint-based Style vs. Supervision-based Synthesis*

**Analysis:** VectorFusion establishes style control by tailoring the primitive types—such as unclosed curves for line art or squares for pixel art—directly within the rendering pipeline, effectively embedding stylistic priors into the geometric representation. In contrast, VectorPainter situates its approach within a broader ecosystem of supervision, highlighting how differentiable rasterization enables the use of CLIP or diffusion models to guide the optimization of these primitives. While VectorFusion emphasizes the specific parameterization of paths and the inherent lossiness of the vectorization process, VectorPainter focuses on the architectural evolution from CLIP-based supervision to generative diffusion-based guidance.

---


### Comparison #9 | C. Ablation: Number of paths

> **VectorFusion excerpt:** C. Ablation: Number of paths VectorFusion optimizes path coordinates and colors, but the number of primitive paths is a non-differentiable hyper-parameter. Vector graphics with fewer paths will be more abstract, whereas photorealism and details can be improved with many paths. In this ablation, we experiment with different number of paths....

**Matching VectorPainter sections:** E. Ablation Studies, B. Stylized SVG Synthesis, A. Stroke Style Extraction

**Criteria:** *Compare how each paper approaches style control and primitive initialization in vector graphics synthesis. Specifically, analyze how VectorFusion uses path reinitialization and raster sample initialization versus how VectorPainter extracts and rearranges vectorized strokes from a reference image using stroke imitation learning and style-preserving losses*

#### ↳ E. Ablation Studies

> **VectorPainter excerpt:** E. Ablation Studies 1) Effect of Imitation Learning Strategy. Our imitation learning strategy aims to ensure that strokes extracted from the reference image authentically capture the desired style. As shown in Fig. 7(a), without this strategy, the extracted strokes inadequately reflect the reference style, resulting in noticeable blank holes in...

**Discrepancy:** 🔴 SIGNIFICANT DISCREPANCY

**Role:** Ablation studies evaluating primitive initialization strategies and style-consistency mechanisms.

**Shared Concepts:** 🤝 *Both frameworks utilize gradient-based optimization of Bézier primitives and conduct ablation studies to validate that informed initialization—whether from raster samples or reference images—outperforms random initialization in terms of visual coherence and semantic fidelity.*

**Key Difference:** 📐 *Heuristic-driven path recycling (VectorFusion) vs. Reference-anchored stroke imitation (VectorPainter).*

**Analysis:** VectorFusion adopts a stochastic approach to primitive management, employing a path reinitialization heuristic to recycle underutilized curves during Score Distillation Sampling (SDS) to maximize caption consistency. In contrast, VectorPainter utilizes a deterministic imitation learning strategy to extract and rearrange strokes from a reference image, prioritizing the preservation of specific stylistic "DNA" over the general abstraction levels explored in VectorFusion’s path-count ablations. While VectorFusion trades off structural rigidity for semantic flexibility by incrementally adding paths to improve R-Precision, VectorPainter implements an Optimal Transport Loss and DDIM inversion to ensure that the initialized strokes do not deviate from the reference style during the refinement process.

#### ↳ B. Stylized SVG Synthesis

> **VectorPainter excerpt:** B. Stylized SVG Synthesis In this step, we adopt an optimization-based pipeline for SVG synthesis, following the prior work [6]–[8]. Initially, an SVG is created in the vector space and subsequently rendered to the pixel space using DiffVG [14]. In the pixel space, losses are computed, then gradients are backpropagated...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Style Control Methodology

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rasterization (DiffVG) within an optimization-based pipeline to refine Bézier primitives. They share a common objective of overcoming the limitations of random initialization by incorporating external priors—VectorFusion via raster-to-vector tracing (LIVE) and VectorPainter via reference-based stroke extraction—to improve synthesis quality and semantic/stylistic alignment.*

**Key Difference:** 📐 *Dynamic Path Re-allocation (VectorFusion) vs. Structural Constraint Preservation (VectorPainter).*

**Analysis:** VectorFusion approaches initialization as a cold-start problem, leveraging raster-sampled paths (LIVE) and a dynamic reinitialization heuristic to maximize path utility and caption consistency. Because VectorFusion treats the number of paths as a non-differentiable hyperparameter, it relies on path density and re-allocation to manage the abstraction-to-photorealism tradeoff. Conversely, VectorPainter shifts the focus toward stylistic continuity, initializing with strokes extracted directly from a reference image and employing a dual-level style-preserving loss. While VectorFusion optimizes for semantic coverage through path re-sampling, VectorPainter utilizes Sinkhorn distance as a stroke-level constraint to minimize geometric deviation from the reference style, trading the fluid re-allocation seen in VectorFusion for rigid structural fidelity.

#### ↳ A. Stroke Style Extraction

> **VectorPainter excerpt:** A. Stroke Style Extraction - 1) ![](figures/fileoutpart11.png) i j i 3 j=1 i i j 3 j=1 P= {p}= {(x, y)}, Stroke Extraction: We define the desired vector graphic S as a collection of n vector strokes S = {Ei}in=1. Specifically, each stroke Ei = {si, ci, wi} is represented...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Style Conditioning Strategy

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable rasterization to optimize Bézier primitives and address the non-differentiable nature of vector topology through specialized initialization strategies. They share a common objective of achieving high-fidelity vector synthesis by aligning primitive attributes (color, geometry, and count) with a target visual or semantic distribution.*

**Key Difference:** 📐 *Stochastic Optimization-heavy (VectorFusion) vs. Deterministic Structural-heavy (VectorPainter)*

**Analysis:** Because VectorFusion relies on Score Distillation Sampling (SDS) from a text-to-image prior, it necessitates a dynamic path reinitialization strategy to prevent primitive stagnation and ensure coverage of the generated manifold. While VectorFusion treats the number of paths as a hyperparameter to be optimized via rejection sampling and re-init, VectorPainter derives its primitive count and initial geometry directly from the topology of a reference image via SLIC superpixel segmentation. VectorPainter trades the generative flexibility of VectorFusion’s diffusion-based initialization for structural fidelity, using imitation learning and MSE-based refinement to ensure that the vectorized strokes strictly adhere to the reference's local color and texture attributes.

---


### Comparison #10 | 5.5. Sketches and line drawings

> **VectorFusion excerpt:** 5.5. Sketches and line drawings Figure 2 includes line drawing samples. VectorFusion produces recognizable and clear sketches from scratch without any image reference, even complex scenes with multiple objects. In addition, it is able to ignore distractor terms irrel- evant to sketches, such as “watercolor” or “Brightly colored” and capture...

**Matching VectorPainter sections:** B. Stylized SVG Synthesis, A. Stroke Style Extraction, B. Comparison Baselines

**Criteria:** *Compare how each paper approaches style control and primitive initialization in vector graphics synthesis. Specifically, analyze how VectorFusion uses path reinitialization and raster sample initialization versus how VectorPainter extracts and rearranges vectorized strokes from a reference image using stroke imitation learning and style-preserving losses*

#### ↳ B. Stylized SVG Synthesis

> **VectorPainter excerpt:** B. Stylized SVG Synthesis In this step, we adopt an optimization-based pipeline for SVG synthesis, following the prior work [6]–[8]. Initially, an SVG is created in the vector space and subsequently rendered to the pixel space using DiffVG [14]. In the pixel space, losses are computed, then gradients are backpropagated...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Optimization Pipeline and Primitive Initialization Strategy

**Shared Concepts:** 🤝 *Both frameworks leverage differentiable rasterization (DiffVG) to bridge the gap between vector primitives and pixel-space loss functions, employing gradient-based optimization to iteratively refine Bézier stroke parameters.*

**Key Difference:** 📐 *Reference-conditioned deterministic initialization vs. Text-conditioned stochastic initialization.*

**Analysis:** While VectorFusion prioritizes zero-shot synthesis from text prompts by utilizing path reinitialization to maintain sketch clarity and semantic alignment, VectorPainter shifts the complexity to the initialization phase by extracting and rearranging vectorized strokes directly from a reference image. Because VectorPainter employs a style-preserving loss based on Optimal Transport (Sinkhorn distance), it enforces structural fidelity to a specific style source, whereas VectorFusion trades this explicit reference-matching for the flexibility of semantic-driven generation, filtering out stylistic distractors through the diffusion prior. Consequently, VectorPainter’s methodology is optimized for style transfer and reconstruction, while VectorFusion’s approach is designed for robust text-to-vector synthesis without the need for external visual anchors.

#### ↳ A. Stroke Style Extraction

> **VectorPainter excerpt:** A. Stroke Style Extraction - 1) ![](figures/fileoutpart11.png) i j i 3 j=1 i i j 3 j=1 P= {p}= {(x, y)}, Stroke Extraction: We define the desired vector graphic S as a collection of n vector strokes S = {Ei}in=1. Specifically, each stroke Ei = {si, ci, wi} is represented...

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Primitive Initialization and Style Conditioning Strategy

**Shared Concepts:** 🤝 *Both frameworks utilize differentiable vector primitives—specifically Bézier curves—and gradient-based optimization to synthesize stylized graphics. They share a common objective of overcoming the non-convexity of vector space by employing strategic initialization techniques to align primitives with a target semantic or visual distribution.*

**Key Difference:** 📐 *Stochastic Generative Sampling vs. Deterministic Structural Extraction*

**Analysis:** Because VectorFusion operates primarily in a text-to-SVG generative context, it relies on stochastic raster sample initialization and periodic path reinitialization to prevent the optimization from collapsing into poor local minima. In contrast, VectorPainter adopts a reconstructive approach where style is explicitly "extracted" from a reference image via SLIC superpixel segmentation, providing a deterministic structural prior that bypasses the need for random sampling. While VectorFusion trades off structural precision for semantic flexibility by ignoring distractor terms in the prompt, VectorPainter prioritizes stylistic fidelity through a dedicated imitation learning phase that optimizes stroke parameters (color, width, and position) against a specific reference image using MSE loss. Consequently, VectorFusion’s pipeline is designed for open-domain synthesis from scratch, whereas VectorPainter’s methodology is optimized for high-fidelity style transfer and structural preservation of existing visual assets.

#### ↳ B. Comparison Baselines

> **VectorPainter excerpt:** B. Comparison Baselines To synthesize stylized vector graphics, there are mainly three approaches: (1) Synthesis though Text Prompt and Reference Image. Like the existing method StyleCLIPDraw [3] and our VectorPainter, these methods generate stylized vector graphics directly based on a given text and a reference image. (2) Ras-terization then Vectorization....

**Discrepancy:** 🟡 MODERATE DIFFERENCE

**Role:** Methodology Positioning and Experimental Scope

**Shared Concepts:** 🤝 *Both frameworks operate within the domain of optimization-based vector graphics synthesis, utilizing differentiable rendering to map semantic descriptors to Bézier-based primitives. They share the objective of producing stylized, recognizable visual outputs—such as sketches or artistic strokes—by iteratively refining primitive parameters to satisfy high-level aesthetic and semantic constraints.*

**Key Difference:** 📐 *Stochastic Text-to-SVG Optimization vs. Reference-Guided Style Transfer*

**Analysis:** Because VectorFusion prioritizes zero-shot synthesis from text prompts, it relies on path reinitialization and raster sample initialization to navigate the non-convex loss landscape of sketch generation without a visual anchor. While VectorFusion demonstrates semantic robustness by filtering out distractor terms to maintain sketch clarity, VectorPainter shifts the architectural focus toward reference-based synthesis, employing stroke imitation learning to extract and rearrange primitives from a source image. Consequently, VectorFusion trades the stylistic precision of a reference image for the flexibility of unconstrained generation, whereas VectorPainter utilizes style-preserving losses to ensure that the synthesized strokes maintain the specific textural and structural DNA of the provided reference.

---

