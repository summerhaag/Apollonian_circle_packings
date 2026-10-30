# Apollonian_circle_packings

Code for computing with and drawing Apollonian circle packings, by [Summer Haag](https://summerhaag.github.io).

## Files

### `DescQuad_class.sage`
A [SageMath](https://www.sagemath.org/) class, `DescartesQuad`, for Descartes quadruples: four curvatures
(a, b, c, d) satisfying (a + b + c + d)² = 2(a² + b² + c² + d²). It can:

- build a quadruple from four curvatures, or from three (the fourth is computed)
- apply Apollonian swaps and words in the Apollonian group
- work over number fields (e.g. ℤ[√2]), including norms and residues of curvatures
- compute the associated binary quadratic forms and parameter-space points
- generate and plot packings (bounded, half-plane, strip, full-plane), with curvature labels
- generate and plot super packings

Load it in Sage:
```python
load("DescQuad_class.sage")
q = DescartesQuad([-1, 2, 2, 3])
q = DescartesQuad([1, 4, 4])   # fourth curvature computed automatically
```

### Notebooks
- `apol_functions.ipynb`: helper functions for Apollonian packings
- `generate_packing_final.ipynb`: generating packing pictures
