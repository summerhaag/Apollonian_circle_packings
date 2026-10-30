import numpy as np

#Graphics functions to help with quickly graphing packings
def circle_in_graphics(G, center, radius, tol=1e-12):
    """
    Check whether a circle is already present in a Sage Graphics object.

    INPUT:
        G      -- Graphics object to search
        center -- (cx, cy) tuple for the circle center
        radius -- radius of the circle
        tol    -- floating-point tolerance for coordinate comparison (default 1e-12)

    OUTPUT:
        True if a Circle primitive with matching center and |radius| exists in G, else False.
    """
    cx, cy = center
    for obj in G._objects:
        if isinstance(obj, sage.plot.circle.Circle):
            x = obj.x
            y = obj.y
            r = obj.r
            if (abs(x - cx) < tol and abs(y - cy) < tol and abs(abs(r) - abs(radius)) < tol):
                return True
    return False

def label_circle_by_center_radius(center, radius, as_fraction=True,fill_factor=0.8, min_fontsize=6, max_fontsize=40,offset_factor=0, **text_opts):
    """
    Label a circle with its curvature (reciprocal of radius), auto-fitting the text inside.

    INPUT:
        center       -- (x, y) tuple for the circle center
        radius       -- radius of the circle; must be > 0
        as_fraction  -- if True, display label as a rational fraction via LaTeX (default True)
        fill_factor  -- fraction of the circle diameter the text should fill (default 0.8)
        min_fontsize -- minimum allowed font size in points (default 6)
        max_fontsize -- maximum allowed font size in points (default 40)
        offset_factor -- shifts the text by offset_factor * radius from center (default 0)
        **text_opts  -- additional keyword arguments passed to Sage's text() function

    OUTPUT:
        A Sage Graphics text object with the curvature label, or an empty Graphics() if radius <= 0.

    NOTATION:
        The label displayed is 1/radius, i.e. the curvature k of the circle.
    """
    if radius <= 0:
        return Graphics()

    x, y = center
    dx = offset_factor * radius
    dy = offset_factor * radius

    # Compute reciprocal
    reciprocal = 1 / radius

    # Format LaTeX label
    if as_fraction:
        from sage.rings.rational_field import QQ
        try:
            reciprocal_label = QQ(reciprocal)
            label_text = f"${latex(reciprocal_label)}$"
        except TypeError:
            label_text = f"${latex(reciprocal)}$"
    else:
        label_text = f"{reciprocal}"

    # Heuristic: count "visible characters" in LaTeX string (ignore $)
    label_stripped = label_text.replace("$","")
    label_length = max(1, len(label_stripped))  # avoid divide by zero

    # Compute fontsize proportional to radius but inversely to string length
    fontsize_est = radius * fill_factor * 30 * (1 / (label_length ** 0.5))

    fontsize = max(min_fontsize, min(max_fontsize, fontsize_est))

    label = text(label_text, (x + dx, y + dy),fontsize=fontsize,horizontal_alignment='center',vertical_alignment='center',**text_opts)
    return label

def _format_curvature(curv):
    """
    Return a display string for a curvature element with no decimals.

    Priority order:
      1. Plain integer  (ZZ, int, or whole-number float/numpy) -> '$7$', '$-2$', etc.
      2. Sage algebraic element  -> '$<latex>$' using Sage's latex(), e.g. '$\\sqrt{2}+3$'
      3. Rational non-integer    -> '$p/q$'
      4. Float that is a simple fraction (limit_denominator 1000) -> '$p/q$'
      5. Fallback                -> '$str(curv)$'

    Every return value is wrapped in '$...$' so matplotlib renders it as mathtext.
    A plain '7' and a mathtext '$\\sqrt{2}+3$' come out in different fonts, which looks
    inconsistent in a packing where some curvatures are rational and some are not, so
    the integers go through mathtext too.
    """
    import re
    # 1. integer check (handles ZZ, int, whole-number numpy/float)
    try:
        n = int(round(float(curv)))
        if abs(float(curv) - n) < 1e-9:
            return '$' + str(n) + '$'
    except (TypeError, ValueError):
        pass

    # 2. Sage algebraic element: use latex() if it contains no decimal points.
    #    A symbolic curvature is latexed exactly as it was written, so a seed entered
    #    as (1+sqrt(2))^2 would be labeled '(\sqrt{2}+1)^{2}' -- several times wider
    #    than the equivalent '2\sqrt{2}+3'. Expand it too and keep whichever form
    #    renders narrower.
    try:
        candidates = [str(latex(curv))]
        try:
            candidates.append(str(latex(curv.expand())))
        except (AttributeError, TypeError, ValueError):
            pass
        candidates = [c for c in candidates if '.' not in c]
        if candidates:
            return '$' + min(candidates, key=lambda c: _label_box_em('$' + c + '$')[0]) + '$'
    except Exception:
        pass

    # 3. Rational non-integer (QQ or Fraction)
    try:
        from sage.rings.rational import Rational
        r = Rational(curv)
        return '$' + str(r.numerator()) + '/' + str(r.denominator()) + '$'
    except Exception:
        pass

    # 4. Float approximation as simple fraction
    try:
        from fractions import Fraction
        frac = Fraction(float(curv)).limit_denominator(1000)
        if abs(float(curv) - float(frac)) < 1e-6:
            if frac.denominator == 1:
                return '$' + str(frac.numerator) + '$'
            return '$' + str(frac.numerator) + '/' + str(frac.denominator) + '$'
    except Exception:
        pass

    return '$' + str(curv) + '$'


def _plain_len(s):
    """Character count of a LaTeX fragment with commands and braces removed (min 1)."""
    import re
    return max(1, len(re.sub(r'\\[a-zA-Z]+|[{}\s]', '', s)))


def _label_box_estimate(s):
    """
    Estimate the rendered bounding box of a (possibly mathtext) label, in em units.

    Returns (width_em, height_em): multiply by the font size in points to get the
    box the string will occupy.  Used to fit labels inside circles -- a label like
    '$2\\sqrt{2}+3$' is several times wider than '$7$' at the same font size, so
    sizing on character count alone (which is what the old estimate did) let long
    algebraic curvatures spill outside their circle.

    Widths are in units of the font size: an ordinary glyph is ~0.6 em wide, a
    superscript ~0.42 em, and a radical sign ~0.55 em.  Height starts at ~0.78 em
    (digit height plus a little slack) and grows for fractions and superscripts.
    Whitespace is dropped because mathtext ignores it.  The estimate only needs to
    be good enough for font scaling, not exact.
    """
    import re
    t = str(s).strip()
    is_math = t.startswith('$') and t.endswith('$')
    t = t.strip('$')
    if not is_math:
        return (0.60 * max(1, len(t)), 0.78)

    t = re.sub(r'\s+', '', t)
    # latex spacing commands: thin spaces render narrow, '\!' pulls text together
    t, n_thin = re.subn(r'\\[,;:]', '', t)
    t = re.sub(r'\\!', '', t)
    width = 0.17 * n_thin
    height = 0.78

    # \frac{a}{b}: as wide as its widest row, and roughly twice as tall
    for num, den in re.findall(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', t):
        width += 0.60 * max(_plain_len(num), _plain_len(den)) + 0.20
        height = max(height, 1.70)
    t = re.sub(r'\\frac\{[^{}]*\}\{[^{}]*\}', '', t)

    # normalize bare '^2' / '_1' to braced form, then size scripts at ~70% width;
    # a script sits above (or below) the body, so it also raises the box
    t = re.sub(r'([\^_])(?!\{)(.)', r'\1{\2}', t)
    scripts = re.findall(r'([\^_])\{([^{}]*)\}', t)
    for kind, body in scripts:
        width += 0.42 * _plain_len(body)
    t = re.sub(r'[\^_]\{[^{}]*\}', '', t)

    # \sqrt{x}: radical sign plus its content, and a taller box for the overbar
    for body in re.findall(r'\\sqrt\{([^{}]*)\}', t):
        width += 0.55 + 0.60 * _plain_len(body)
        height = max(height, 1.00)
    t = re.sub(r'\\sqrt\{[^{}]*\}', '', t)

    # \left / \right only resize the delimiter that follows; they draw nothing
    t = re.sub(r'\\(left|right|big{1,2}[lr]?)(?![a-zA-Z])', '', t)

    # any remaining LaTeX command renders as roughly one glyph
    t, n_cmd = re.subn(r'\\[a-zA-Z]+', '', t)
    width += 0.60 * n_cmd

    # what is left is ordinary glyphs (braces are not rendered); binary
    # operators pick up thin space on either side in mathtext
    t = re.sub(r'[{}]', '', t)
    width += 0.60 * len(t) + 0.16 * len(re.findall(r'[+\-=]', t))

    # delimiters stretch to cover whatever they wrap, so they push the box taller
    if re.search(r'[()\[\]]', t):
        height *= 1.30
    # a superscript rides above the body, a subscript hangs below
    if any(kind == '^' for kind, _ in scripts):
        height += 0.30
    if any(kind == '_' for kind, _ in scripts):
        height += 0.20

    return (max(0.60, width), height)


#measured label boxes, keyed by label string.  A packing has far fewer distinct
#labels than circles, so measuring each string once keeps labeling cheap.
_LABEL_BOX_CACHE = {}


def _label_box_measured(s):
    """
    Measure the ink box of a (possibly mathtext) label, in em units, via TextPath.

    Returns (width_em, height_em) of the actual glyph outlines matplotlib will draw,
    or None if the string cannot be measured (bad mathtext, missing fonts, ...).
    This is exact where the analytic estimate is not: mathtext puts wide spacing
    around binary operators and draws a radical sign much wider than a digit, so
    '$2\\,\\sqrt{2}+3$' is about 5 em wide, not the ~3.3 em a per-glyph estimate gives.
    """
    try:
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties
        size = 100.0                      # measure big, then divide, for precision
        bb = TextPath((0, 0), str(s), size=size, prop=FontProperties()).get_extents()
        if bb.width <= 0 or bb.height <= 0:
            return None
        return (bb.width / size, bb.height / size)
    except Exception:
        return None


def _label_box_em(s):
    """
    Return (width_em, height_em) for a label string, measured if possible.

    Falls back to _label_box_estimate() when the string cannot be measured.
    Results are cached in _LABEL_BOX_CACHE, so the cost is one measurement per
    distinct label rather than per circle.
    """
    key = str(s)
    box = _LABEL_BOX_CACHE.get(key)
    if box is None:
        box = _label_box_measured(key) or _label_box_estimate(key)
        _LABEL_BOX_CACHE[key] = box
    return box


def keep_only_circles(G):
    """
    Return a new Graphics object containing only circle and line primitives from G.

    INPUT:
        G -- a Sage Graphics object

    OUTPUT:
        A new Graphics object with all non-circle, non-line primitives removed.
        Useful for stripping text labels or other overlays before further manipulation.
    """
    newG = Graphics()
    candidates = getattr(G, "_objects", [])

    for entry in candidates:
        # Each entry is typically (primitive, options)
        prim = entry[0] if isinstance(entry, tuple) else entry

        clsname = prim.__class__.__name__

        # Keep if it's a circle or a line
        if (clsname in ("Circle", "Line") or hasattr(prim, "radius") and hasattr(prim, "center") or hasattr(prim, "points")):
            newG.add_primitive(prim)

    #a fresh Graphics() loses the aspect_ratio=1 that circle() sets, which squashes circles into ellipses
    newG.set_aspect_ratio(1)
    return newG


class DescartesQuad:
    """
    A quadruple (a, b, c, d) of curvatures satisfying the Descartes Circle Theorem:

        (a + b + c + d)^2 = 2*(a^2 + b^2 + c^2 + d^2)

    Each entry is the curvature (reciprocal of radius) of a circle in an Apollonian
    circle packing.  Negative curvature means the other three circles are internally
    tangent to that circle (the outer enclosing circle).

    Accepts flat array [a, b, c, d] or array [a, b, c] and finds d = a+b+c - 2*sqrt(ab+ac+bc).

    ATTRIBUTES:
        _entries      -- list [a, b, c, d] of curvatures
        _desc_matrix  -- 4x4 Descartes matrix with columns [k, k*Re(z), k*Im(z), k'] per circle
        _circles      -- list of [x_center, y_center, radius] for all generated circles
        _graph        -- Sage Graphics object of the current packing
        _supercircles -- circles generated by the super-Apollonian group
        _field        -- number field the entries live in (default RR)
        _forms        -- associated binary quadratic forms (BQFs)
        _points       -- parameter-space points associated to each BQF

    EXAMPLES::

        sage: q = DescartesQuad([-1, 2, 2, 3])
        sage: q = DescartesQuad([1, 4, 4])   # fourth curvature computed automatically
    """
#######################
#BASIC INTIALIZATIONS
#######################
    def __init__(self, x):
        """
        Initialize a DescartesQuad from curvatures.

        INPUT:
            x -- list of 4 curvatures [a, b, c, d] satisfying the Descartes equation
                    (a+b+c+d)^2 = 2*(a^2+b^2+c^2+d^2),
                 OR a list of 3 curvatures [a, b, c] from which the fourth is computed as
                    d = a + b + c - 2*sqrt(a*b + a*c + b*c).

        Raises TypeError if the input is the wrong length or does not satisfy the equation.

        EXAMPLES::

            sage: q = DescartesQuad([-1, 2, 2, 3])
            sage: q = DescartesQuad([1, 4, 4])
        """
        quad=self._normalize(x)
        self._entries=quad

        #if the starting quadruple is made of rational integers, every curvature
        #generated by the Apollonian group is also a rational integer since the
        #group matrices are integral. So norms should just be the curvature
        #itself, not N(k)=a^2-d*b^2 computed after coercion into a quadratic ring.
        self._is_integral=self._check_integral(quad)

        #compute the desc matrix
        A,B,C,Dd=self._entries
        self._centers=self._center_circles(A,B,C,Dd)
        cc=self._center_cocurv(A,B,C,Dd)

        self._desc_matrix=matrix(cc).transpose()

        #compute circles in packing for graphing purposes
        self._circles=self._isolate_circle()

        #graphics for packing
        self._graph=self._graph_circle()

        #only for generating the superpacking and graphing superpackings
        self._supercircles=self._isolate_circle()
        self._super_packing_graph=self._graph_circle('black')

        #curvatures parallel to _circles, used for norm labeling
        #_entries and _circles are in the same column order, so _entries[i] == curvature for _circles[i]
        self._curvatures=list(self._entries)

        #integer norms parallel to _circles/_curvatures, cached so rendering never calls .norm() again
        self._norms=[self._curv_to_norm(k) for k in self._curvatures]

        #Field we are in
        #assumption is just real numbers
        self._field=RR

        #BQFs associated with the packing
        self._forms=self._form
        #parameter space points
        self._points=self._point

    def _normalize(self,x):
        """
        Validate and normalize the input into a Descartes quadruple [a, b, c, d].

        INPUT:
            x -- iterable of length 3 or 4

        OUTPUT:
            List [a, b, c, d] satisfying the Descartes equation.
            If length 3 is given, computes d = a+b+c - 2*sqrt(a*b+a*c+b*c).
            If two or more entries are negative, negates all three before computing d.

        Raises TypeError if the input cannot be coerced to a valid quadruple.
        """
        try:
            lisX=list(x)
            if len(lisX)==4: #check if Descartes Quad
                a,b,c,d=lisX
                lhs=(a+b+c+d)^2
                rhs=2*(a^2+b^2+c^2+d^2)
                if lhs==rhs:
                    return [a,b,c,d]
                #floats can fail the exact check above by ~1e-14 of rounding error;
                #fall back to a numeric tolerance so float-based construction (used
                #to keep _center_circles fast -- see set_field's docstring) validates
                if abs(float(lhs)-float(rhs))<1e-6:
                    return [a,b,c,d]
            if len(lisX)==3: #find missing curv
                a,b,c=lisX
                #checking negatives-- if two negatives or three negatives just negate all the numbers
                neg=[1 for n in lisX if n<0]
                if len(neg)>1:
                    a=-a
                    b=-b
                    c=-c
                d=a+b+c-2*sqrt(a*b+a*c+b*c)
                return [a,b,c,d]
        except (ValueError, TypeError) as e:
                # wrong lengths or non-iterables at some level
                # by just passing, we end up sending all errors to the last TypeError
                pass
        # if we get here, something went wrong so give a general error
        raise TypeError("Expected 4 entries (flat) satisfying Desc Eq or 3 entries (flat)")

    #how it prints out
    def __repr__(self):
        a,b,c,d = self._entries
        return f"({a}, {b}, {c}, {d})"
###############################
#INTEGRAL AND LG CONSIDERATIONS
###############################
    def _check_integral(self, quad):
        """
        Check whether every entry of a quadruple is a rational integer.

        OUTPUT:
            True if QQ(k) is an integer for every k in quad, else False
            (includes the case where k can't be coerced to QQ at all, e.g. a
            genuine quadratic irrational).
        """
        try:
            return all(QQ(k) in ZZ for k in quad)
        except (TypeError, ValueError):
            return False

    def _curv_to_norm(self, curv):
        """
        Compute the norm of a curvature element and return it as a Python int where possible.

        If the packing is integral (self._is_integral), every curvature is a rational
        integer and its norm is just itself -- calling .norm() would instead compute
        a^2 - d*b^2 on the quadratic-ring coercion of an integer (b=0), which squares
        the value instead of returning it unchanged.

        Otherwise, for elements k of a quadratic ring, returns k.norm() (= a^2 - d*b^2
        for Z[sqrt(d)]). For numpy/float scalars that are whole numbers, returns the
        integer value. Falls back to returning curv unchanged if conversion fails.
        """
        if self._is_integral:
            try:
                return int(round(float(curv)))
            except (TypeError, ValueError):
                return curv
        try:
            n = curv.norm()
        except AttributeError:
            n = curv
        try:
            int_val = int(round(float(n)))
            if abs(float(n) - int_val) < 1e-6:
                return int_val
        except (TypeError, ValueError):
            pass
        return n

    #see which residue classes mod m actually occur among the generated norms --
    #useful for hunting local-obstruction patterns analogous to the classical
    #mod-24 restriction on curvatures of integral Apollonian packings
    def residues_mod(self, m=24):
        """
        Return the sorted set of distinct residues mod m among self._norms.

        INPUT:
            m -- modulus to reduce norms by (default 24)

        OUTPUT:
            Sorted list of distinct values of N(k) mod m over all currently
            generated circles. Call after a *_generate_circles method so that
            self._norms actually has more than the initial 4 entries.

        EXAMPLE::

            sage: q.bounded_generate_circles(500)
            sage: q.residues_mod(24)
            [0, 4, 7, 12, 15, 16]
        """
        return sorted(set(int(n) % m for n in self._norms))
##########################
#RING AND FIELD FUNCTION
##########################
    #if you want to change the field/ring of integers
    #helpful for quadratic things
    def set_field(self, field=QQ, exact_curvatures=None):
        """
        Change the number field (or ring) that the curvature entries are coerced into.

        INPUT:
            field            -- a Sage field or number field (default QQ).
                                Pass RR to revert to real-number mode without coercion.
            exact_curvatures -- optional list of 4 elements already in field's ring of integers.
                                Use this when self._entries holds float approximations: float
                                initialization keeps the geometry working but loses exact values
                                needed for norm labels.  If None, coerces self._entries directly.

        Use this when working with quadratic fields, e.g.::

            sage: K.<a> = QuadraticField(2)
            sage: q = DescartesQuad([-2.0, 5*float(sqrt(2))-3, float(sqrt(2))+3, 6*float(sqrt(2))-4])
            sage: R = K.ring_of_integers()
            sage: q.set_field(K, exact_curvatures=[R(-2), -3+5*a, 3+a, -4+6*a])

        Raises TypeError if coercion fails.
        """
        if field!=RR:
            try:
                ring=field.ring_of_integers()
                if exact_curvatures is not None:
                    quad=[ring(x) for x in exact_curvatures]
                else:
                    quad=[ring(x) for x in self._entries]
            except Exception as e:
                raise TypeError("Coercion to field or integer ring failed.") from e
            self._entries=quad
            #update the initial 4 curvatures and norms to exact ring elements
            self._curvatures[:4]=list(quad)
            self._norms[:4]=[self._curv_to_norm(k) for k in quad]
            self._field=field
        if field==RR:
            self._field=RR


##############################################    
#APOLLONIAN GROUP AND SUPER APOLLONIAN SWAPS
##############################################
    #for having the apollonian group act on the quadruple


    #swap the indicated circle for the other soddy circle
    #does not change our value
    def swap(self, index):
        """
        Apply the Apollonian swap matrix S_{index+1} to the current quadruple.

        Each generator S_n of the Apollonian group replaces the n-th circle with
        the other Soddy circle tangent to the remaining three, via the formula:
            new_k_n = 2*(k_1 + k_2 + k_3) - k_n   (where the others are fixed).

        The matrices used are:
            S_1: [[-1,0,0,0],[2,1,0,0],[2,0,1,0],[2,0,0,1]]
            S_2: [[1,2,0,0],[0,-1,0,0],[0,2,1,0],[0,2,0,1]]
            S_3: [[1,0,2,0],[0,1,2,0],[0,0,-1,0],[0,0,2,1]]
            S_4: [[1,0,0,2],[0,1,0,2],[0,0,1,2],[0,0,0,-1]]

        INPUT:
            index -- integer 0, 1, 2, or 3 indicating which circle to swap out

        OUTPUT:
            (quad, descMatrix) -- tuple of the new curvature list and updated Descartes matrix.
            Does NOT modify self in place; use swap_change() for that.
        """
        quad=vector(self._entries)
        descMatrix=self._desc_matrix
        Sn=matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        if index+1==1:
            Sn=matrix([[-1,0,0,0],[2,1,0,0],[2,0,1,0],[2,0,0,1]])
        if index+1==2:
            Sn= matrix([[1,2,0,0],[0,-1,0,0],[0,2,1,0],[0,2,0,1]])
        if index+1==3:
            Sn= matrix([[1,0,2,0],[0,1,2,0],[0,0,-1,0],[0,0,2,1]])
        if index+1==4:
            Sn= matrix([[1,0,0,2],[0,1,0,2],[0,0,1,2],[0,0,0,-1]])
        quad=list(quad*Sn)
        descMatrix=descMatrix*Sn
        return quad, descMatrix

    #apply a word to the quadruple and add those circles to the packing
    def swap_word(self, word):
        """
        Apply a sequence of Apollonian swaps and add each new circle to the packing.

        INPUT:
            word -- list of indices in {0,1,2,3}, e.g. [0, 2, 1, 3].
                    Each index is passed to _swap_add(), which calls swap() and appends
                    the resulting circle to self._circles.

        Modifies self._circles and self._entries in place.

        EXAMPLE::

            sage: q.swap_word([0, 2, 1])
        """
        for i in range(0,len(word)):
            self._swap_add(word[i])

    #apply a word to the quadruple
    def swap_word_quad(self, word):
        """
        Apply a sequence of Apollonian swaps, updating the quadruple only (no circle added).

        INPUT:
            word -- list of indices in {0,1,2,3}.
                    Each swap changes self._entries and self._desc_matrix but does NOT
                    append to self._circles.

        Use this to navigate the Apollonian graph without generating packing circles.

        EXAMPLE::

            sage: q.swap_word_quad([1, 0, 3])
        """
        for i in range(0,len(word)):
            self.swap_change(word[i])

    #swap soddy circle-changing quadruple-and add it to the raw circles
    def _swap_add_norm(self,index):
        """
        Swap the circle at the given index, update self in place, and append the new circle.

        Applies swap(index) to get the new quadruple and Descartes matrix, extracts the
        new circle's [x_center, y_center, radius] from the matrix column, and appends it
        to self._circles if not already present.  Also appends the curvature to
        self._curvatures so that norm labels remain aligned with self._circles.

        INPUT:
            index -- integer 0--3 indicating which circle to replace
        """
        V,M=self.swap(index)
        circles=self._circles
        curvatures=self._curvatures
        norms=self._norms
        p=M[0][index]
        r=M[1][index]
        s=M[2][index]
        q=M[3][index]
        if p!=0:
            newcirc=([r/p,s/p, 1/p])
        if p==0:
            newcirc=([r,q/2,p])
        if newcirc not in circles:
            circles.append(newcirc)
            curv=V[index]  # exact ring element after set_field
            curvatures.append(curv)
            norms.append(self._curv_to_norm(curv))  # cache norm as int; free at render time
        self._circles=circles
        self._curvatures=curvatures
        self._norms=norms
        self._entries=V
        self._desc_matrix=M
        self._forms=self._form



    #SUPER apollonian group

    def super_swap(self, index):
        """
        Apply a generator of the super-Apollonian group to the current quadruple.

        The super-Apollonian group is generated by 8 matrices: the 4 standard Apollonian
        generators S_1..S_4 (index 0--3) and their transposes S_1^T..S_4^T (index 4--7).
        The transpose generators act on the *rows* of the Descartes matrix rather than the
        columns, producing the dual / super packing.

        INPUT:
            index -- integer 0--7.
                     0--3: delegates to self.swap(index) (standard Apollonian group).
                     4--7: applies the transpose of S_{index-3}.

        OUTPUT:
            (quad, descMatrix) -- new curvature list and Descartes matrix. Does not modify self.
        """
        quad=vector(self._entries)
        descMatrix=self._desc_matrix
        Sn=matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        if 0<=index<=3:
            return self.swap(index)
        elif index+1==5:
            Sn=(matrix([[-1,0,0,0],[2,1,0,0],[2,0,1,0],[2,0,0,1]])).transpose()
        elif index+1==6:
            Sn= (matrix([[1,2,0,0],[0,-1,0,0],[0,2,1,0],[0,2,0,1]])).transpose()
        elif index+1==7:
            Sn= (matrix([[1,0,2,0],[0,1,2,0],[0,0,-1,0],[0,0,2,1]])).transpose()
        elif index+1==8:
            Sn= (matrix([[1,0,0,2],[0,1,0,2],[0,0,1,2],[0,0,0,-1]])).transpose()
        quad=list(quad*Sn)
        descMatrix=descMatrix*Sn
        return quad, descMatrix

    #swap soddy circle-changing quadruple-and add it to the raw supercircles
    def _super_swap_add(self,index):
        """
        Apply a super-Apollonian swap, update self, and append the new circle to self._supercircles.

        INPUT:
            index -- integer 0--7 for the super-Apollonian generator to apply.
                     Calls swap_change(index, grp=8) then extracts the new circles via
                     _isolate_circle() and merges them into self._supercircles.
        """
        oldCircles=self._supercircles
        self.swap_change(index,8)
        circles=self._isolate_circle()
        #dont add circles already there
        newCirc=oldCircles +[x for x in circles if x not in oldCircles]
        self._supercircles=newCirc


    #making the modes so only one function is necessary
    #default is to use the usual group
    def _swap_add(self, index, grp=4):
        """
        Dispatch to the correct swap-and-add routine based on which group is in use.

        INPUT:
            index -- circle index to swap
            grp   -- 4 for the standard Apollonian group (default), 8 for super-Apollonian.
        """
        if grp==4:
            self._swap_add_norm(index)
        else:
            self._super_swap_add(index)

    #actually changes self to the swapped version
    def swap_change(self,index, grp=4):
        """
        Apply a swap in place, updating self._entries and self._desc_matrix.

        Unlike swap() which only returns new values, this method mutates self.

        INPUT:
            index -- integer indicating which circle to swap (0--3 for grp=4, 0--7 for grp=8)
            grp   -- 4 for standard Apollonian group (default), 8 for super-Apollonian group.

        EXAMPLE::

            sage: q.swap_change(0)   # replace circle 0 with its other Soddy partner
        """
        if grp==4:
            swapped=self.swap(index)
            self._entries=swapped[0]
            self._desc_matrix=swapped[1]
            self._forms=self._form
        else:
            swapped=self.super_swap(index)
            self._entries=swapped[0]
            self._desc_matrix=swapped[1]

####################
#PARAMETER SPACE
#####################
    #ACP to BQF
    def _form(self):
        """
        Compute the binary quadratic forms (BQFs) associated to the Apollonian circle packing.

        For each index n in {0,1,2,3}, produces the BQF with coefficients
            [k_n,  k_n + k_{n+1},  k_n + k_{n+1} + k_{n+2} - k_{n+3},  k_n + k_{n+2}]
        (indices taken mod 4).

        OUTPUT:
            List of 4 BQFs, each represented as a list of 4 coefficients [A, B, C, D].

        NOTATION:
            k_0, k_1, k_2, k_3 = self._entries (curvatures).
            The BQF for index n encodes the arithmetic of circles tangent to circle n.
        """
        V=self._entries
        BQF=[]
        for index in range(0,4):
            n=V[index]
            a=V[index+1%4]
            b=V[index+2%4]
            c=V[index+3%4]
            BQF.append([n, n+a,n+a+b-c,n+b])
        return BQF

    #associated point for parameter space
    def _point(self):
        """
        Compute the parameter-space point associated to each BQF in self._forms.

        For BQF [A, B, C, D] the point is (-C / (2*B), A / B) when B != 0,
        or an empty list if B = 0.

        OUTPUT:
            List of 4 points (each a [x, y] pair or []) corresponding to self._forms.

        NOTATION:
            Points live in a parameter space where each ACP circle maps to a point
            that tracks the BQF orbit under the Apollonian group.
        """
        BQF=self._forms
        points=[]
        for i in range(0,4):
            bqf=BQF[i]
            if bqf[1]==0:
                points.append([])
            else:
                points.append([-bqf[2]/(2*bqf[1]), bqf[0]/bqf[1]])
        return points



    #DESCARTES MATRIX
    #the following code is for generating the Descartes Matrix
    #which has the co-curv, and curv-center for each circle in the quadruple


#########################
#BUILDING MATRIX OF CURV, CURV-CENTER, CO-CURV
##########################
    #find the centers of the circles
    def _center_circles(self, A,B,C,Dd):
        """
        Compute the (curvature, x-center, y-center) triple for each of the 4 circles.

        Places the initial quadruple in a canonical position:
          - If two curvatures are 0: two lines and two circles in a strip.
          - If one curvature is 0:   one line and three mutually tangent circles.
          - If all nonzero:          the smallest-curvature circle centered at the origin,
                                     the others placed using tangency and the distance formula.
        The fourth center is recovered from the complex Descartes equation.

        INPUT:
            A, B, C, Dd -- the four curvatures (floats or symbolic)

        OUTPUT:
            List of 4 entries, each [k, x_center, y_center], in the same order as the input.
        """
        D=[A,B,C,Dd]
        #sort and find permutations
        D_array=np.array(D)
        perm = np.argsort(D_array)     # get permutation
        D_sorted = D_array[perm]       # apply permutation to D_array
        # build inverse permutation
        inverse_perm = [0] * len(perm)
        for sorted_idx, orig_idx in enumerate(perm):
            inverse_perm[orig_idx] = sorted_idx

        centers=[]
        x, y, z= var('x, y, z')

        #need to find zero curvatures and nonzero curvatures
        zeroes=[1 for curv in D if curv==0]
        nonzeroes=[curv for curv in D if curv!=0]
        #need to know the smallest curvature
        sortednonzeroes=sorted(nonzeroes)

        if len(zeroes)==2: #we have two zeroes
            centers=[[0,0,-1],[0,0,1], [nonzeroes[0], 0, 1/nonzeroes[0]], [nonzeroes[1], 2/nonzeroes[1], 1/nonzeroes[1]]]

        if len(zeroes)==1:
            #so long as 0 is in packing these be the centers
            centers= [[0,0,-1],[sortednonzeroes[0], 0, 1/sortednonzeroes[0]],
                        [sortednonzeroes[1], 2/(sqrt(sortednonzeroes[0]*sortednonzeroes[1])), 1/sortednonzeroes[1]],
                        [sortednonzeroes[2], 2/(sqrt(sortednonzeroes[0]*sortednonzeroes[2])), 1/sortednonzeroes[2]]]
        if (D[0]*D[1]*D[2]*D[3])!=0:
            if sortednonzeroes[0]<0:
                b=sortednonzeroes[1]
                c=sortednonzeroes[2]
                d=sortednonzeroes[3]
            elif sortednonzeroes[0]>0:
                b=sortednonzeroes[0]
                c=sortednonzeroes[1]
                d=sortednonzeroes[2]
            xc=2/(sqrt(b*c)) #the x-value of the center for the circle with curv c
            centers=[[b,0,1/b],[c,xc,1/c]]

            #find the center of the circle with curvature d using the centers of b and c and the fact that it is
            #tangent to both of those circles and the distance formula

            distCenter=sqrt((xc)^2+(1/b-1/c)^2)
            bcDist=1/c-1/b
            x_local=((1/d+1/b)^2-(1/c+1/d)^2+(xc)^2+(1/b-1/c)^2)/(2*distCenter)
            y_local = sqrt(max(0.0, (1/d+1/b)^2 - x_local^2))
            xd=x_local*(xc/distCenter)+y_local*(bcDist/distCenter)
            yd=1/b+x_local*(bcDist/distCenter)-y_local*(xc/distCenter)
            dCenter=[d, xd, yd]

            centers.append(dCenter)
            #use equation from complex desc eq to find last center
            sum_of_curv=D[0]+D[1]+D[2]+D[3]
            a=sum_of_curv-b-c-d
            re_kz=0; im_kz=0; re_k2z=0; im_k2z=0
            for i in range(0,len(centers)):
                re_kz+=centers[i][0]*centers[i][1]
                im_kz+=centers[i][0]*centers[i][2]
                re_k2z+=centers[i][0]^2*centers[i][1]
                im_k2z+=centers[i][0]^2*centers[i][2]
            denom=a^2-(1/2)*sum_of_curv*a
            re_zz=((1/2)*sum_of_curv*re_kz-re_k2z)/denom
            im_zz=((1/2)*sum_of_curv*im_kz-im_k2z)/denom
            aCenter=[a, re_zz, im_zz]
            centers.append(aCenter)

        #sort the centers by curvature
        sorted_centers=sorted(centers, key=lambda X: X[0])
        array_sorted_center=np.array(sorted_centers)
        #now use the inverse permutation to undo and order of the original input
        center_orig_order=array_sorted_center[inverse_perm]
        return list(center_orig_order);

    #find co-curv
    #returns the vector with entries [curv, real-curv-center, imag-curv-center, cocurv] for each curv
    def _center_cocurv(self, A,B,C,Dd):
        """
        Compute the co-curvature for each circle and assemble the Descartes matrix columns.

        For a circle with curvature k and center z = x + i*y, the Descartes matrix column is
            [k,  k*x,  k*y,  k']
        where k' is the co-curvature defined by  k'^2 = k^2*(x^2+y^2) - 1  (up to sign / normalization).

        Special cases:
          - k = 0 (line), first occurrence: co-curvature set to 0.
          - k = 0 (line), second occurrence: co-curvature set to 2*k_2 (from sorted curvatures).

        INPUT:
            A, B, C, Dd -- the four curvatures

        OUTPUT:
            List of 4 entries [k, k*x, k*y, k'] ready to be stacked as columns of the Descartes matrix.
        """
        curv=[A,B,C,Dd]
        #centers=self._center_circles(A,B,C,Dd)
        centers=self._centers
        center_cocurv=[]
        count=0
        for k in range(0,len(centers)):
            V=centers[k]
            p=V[0]
            r=V[1]*V[0]
            s=V[2]*V[0]
            if p==0 and count==0:
                center_cocurv.append([p, V[1], V[2], 0])
                count+=1
            elif p==0 and count==1:
                #strip between y=0 and y=2/k (k = the two equal nonzero curvatures),
                #and a line's co-curv is 2*(height), so co-curv = 4/k
                sorted_curv=sorted(curv)
                center_cocurv.append([p,0,1,4/sorted_curv[2]])
            else:
                q=(r^2+s^2-1)*1/p
                center_cocurv.append([p,r,s,q])
        return center_cocurv

    #change center of indicted circle to new center (x,y)
    def change_center(self,index,x=0,y=0):
        """
        Translate the entire packing so that the circle at the given index has center (x, y).

        Computes the shift (dx, dy) = (x - x_current, y - y_current) for the specified circle
        and applies it uniformly to all circles in self._centers, self._desc_matrix, and
        self._circles.  Lines (curvature 0) are shifted only in y.

        INPUT:
            index -- integer 0--3 indicating which circle to anchor at (x, y)
            x     -- desired x-coordinate of that circle's center (default 0)
            y     -- desired y-coordinate of that circle's center (default 0)

        Modifies self._centers, self._desc_matrix, and self._circles in place.
        Has no effect if the specified circle has curvature 0.
        """
        D=self._desc_matrix
        A,B,C,Dd=self._entries
        #shift all the centers
        if D[0][index]!=0:
            xcircle=D[1][index]/D[0][index]#current x-coord of specified circle center
            ycircle=D[2][index]/D[0][index]#current y-coord of specified circle center
            newCircles=[]
            for i in range(0,4):
                if D[0][i]!=0:
                    curv_center=[D[0][i],(D[1][i]/D[0][i])+x-xcircle, (D[2][i]/D[0][i])+y-ycircle]
                    newCircles.append(curv_center)
                elif D[0][i]==0:
                    curv_center=[0,0,-1]
                    newCircles.append(curv_center)
            #now calculate the new cocurv
            self._centers=newCircles
            cc=self._center_cocurv(A,B,C,Dd)
            self._desc_matrix=matrix(cc).transpose()
            #shift all circles that have been generated
            allCircles=self._circles
            for i in range(0,len(allCircles)):
                if allCircles[i][2]==0:
                    allCircles[i]=[allCircles[i][0], allCircles[i][1]+y-ycircle, allCircles[i][2]]
                elif allCircles[i][2]!=0:
                     allCircles[i]=[allCircles[i][0]+x-xcircle, allCircles[i][1]+y-ycircle, allCircles[i][2]]
    #change the Descartes matrix to a new one
    def change_descMatrix(M, transpose=False):
        """
        Change the Descartes matrix to a new one, updating self._entries and self._centers.

        INPUT:
            M -- 4x4 matrix representing the new Descartes configuration.
            transpose -- boolean indicating whether to transpose the matrix (default False)
        """
        if transpose:
            M = M.transpose()
        quad=sorted(self._entries)
        newQuad=sorted([M[0][0], M[0][1], M[0][2], M[0][3]])
        if quad!=newQuad:
            raise ValueError("New Descartes matrix does not correspond to the same curvatures.")
        self._desc_matrix = M
        self._entries = [M[0][0], M[0][1], M[0][2], M[0][3]]

#####################
#REDUCTION
#################
    #outputs the swapped curv size compared to the bound
    #1 is swapped curv is smaller, -1 is larger, 0 is same size
    #default index is 0 and default bound is 0
    def _swap_size(self, index, bound=0,grp=4):
        """
        Compare the curvature after swapping index against a bound.

        INPUT:
            index -- integer 0--3, the circle to (hypothetically) swap
            bound -- numeric threshold to compare against (default 0)
            grp   -- unused here; present for signature compatibility (default 4)

        OUTPUT:
             1  if swapped curvature < bound  (swap makes it smaller)
             0  if swapped curvature == bound
            -1  if swapped curvature > bound  (swap makes it larger)
        """
        swapD=self.swap(index)[0]
        if swapD[index]< bound:
            return 1
        if swapD[index]==bound:
            return 0
        if swapD[index]>bound:
            return -1

    #given quadruple reduce it to the base quadruple
    #not for unbounded quadruples
    def _reduce_bounded(self):
        """
        Reduce the current quadruple to its canonical (minimal) form for bounded packings.

        Iterates over indices 0--3; whenever a swap produces a smaller curvature than the
        current value at that index, applies the swap and restarts.  Continues until no
        further reductions are possible (analogous to BQF reduction).

        Only valid for packings where all curvatures are positive (bounded / finite packings).
        Updates self._circles, self._graph, self._forms after reduction.
        """
        swap=0
        while swap<=3:
            D=self._entries
            if self._swap_size(swap,D[swap])==1:
                self.swap_change(swap)
                swap=-1
            swap+=1
        #add any new circles
        circles=self._isolate_circle()
        oldCircles=self._circles
        #dont add circles already there
        newCirc=oldCircles +[x for x in circles if x not in oldCircles]
        self._circles=newCirc
        self._graph=self._graph_circle()
        self._forms=self._form

    #given given quadruple with zero curvature reduce to 0-reduction
    #only for quadratic
    def _reduce_0(self):
        """
        Reduce a quadruple containing a zero curvature to its canonical 0-reduced form.

        For quadratic number fields only.  Produces a quadruple of the form
            [0, A^2, B^2, (A+B)^2]
        where A, B are chosen to minimize the leading coefficient, analogous to
        Gauss reduction of binary quadratic forms.

        Requires self._field to be a quadratic number field (set via set_field()).
        Uses the global variable alpha (the generator of the quadratic field).
        Updates self._entries and self._forms.
        """
        V=self._entries
        #need it of the form 0,a^2,b^2,(a+b)^2
        K=self._field
        v=sorted(V)
        a=sqrt(K(v[1]))
        b=sqrt(K(v[2]))
        a0=(1/2)*(a.trace())
        b0=(1/2)*(b.trace())
        a1=(a-a0)/alpha
        b1=(b-b0)/alpha
        G=xgcd(a1,b1)
        g=G[0]
        m=-G[1]
        n=G[2]
        b2=b1/g
        a2=a1/g
        A=a0*b2-a2*b0
        c=b0*n-a0*m+g*alpha
        if A<0:
            A=-A
        if c<0:
            c=-c
        #need to find the biggest curv tangent to 0 and A^2
        x=round(c/A)
        B=(A*x-(b0*n-a0*m + g*sqrt(d)))
        if B<0:
            B=-B
        self._entries=[0,A^2,B^2,(A+B)^2]
        self._forms=self._form




##########################################
#GRAPHING PACKINGS
#######################################

    #input descarte quad and output circle vector of [x-center, y-center, radius] for all 4 circles
    def _isolate_circle(self):
        """
        Extract the geometric circle data from the current Descartes matrix.

        For each column of self._desc_matrix with entries [k, k*x, k*y, k']:
          - If k != 0: circle with center (k*x/k, k*y/k) and radius 1/k.
          - If k  = 0: line represented as [k*x, k', 0] (radius 0 flags a line).

        OUTPUT:
            List of 4 entries, each [x_center, y_center, radius].
            A radius of 0 indicates a straight line (infinite radius).
        """
        M=self._desc_matrix
        circles=[]
        for i in range(0,4):
            p=M[0][i]
            r=M[1][i]
            s=M[2][i]
            if p!=0:
                circles.append([r/p,s/p, 1/p])
            if p==0:
                q=M[3][i]
                circles.append([r,q/2,p]) #line height is co-curv/2, same as _swap_add_norm
        return circles;

    #put the circles generated no in the graph so far into the graph,
    #xLbound and xUbound give a bound on the straight line if applicable
    #changes the graph of the circles
    #label the circles with curvature if label=1 default, and not otherwise
    def graph_circles(self, label=0, colord='black',xLbound=-1, xUbound=1):
        """
        Render all circles in self._circles into self._graph.

        Circles already present in the Graphics object are skipped (no duplicates).
        Lines (radius 0) are drawn as horizontal line segments between xLbound and xUbound.

        INPUT:
            label   -- 0 to draw circles only (default); nonzero to add curvature labels
                       inside each circle via label_circle_by_center_radius().
            colord  -- edge color for circles (default 'black')
            xLbound -- left x-endpoint for drawing lines (default -1)
            xUbound -- right x-endpoint for drawing lines (default 1)

        Modifies self._graph and self._super_packing_graph in place.

        EXAMPLE::

            sage: q.bounded_generate_circles(20)
            sage: q.graph_circles(label=1)
            sage: q._graph.show()
        """
        V=self._circles
        xmin=min([min(V[i][0]+V[i][2],V[i][0]-V[i][2]) for i in range(len(V))])
        xmax=max([max(V[i][0]+V[i][2],V[i][0]-V[i][2])for i in range(len(V))])
        G=self._graph
        for i in range(0,len(V)):
            if V[i][2]==0:
                G+=line([(xmin,V[i][1]), (xmax, V[i][1])], color=colord)
            elif V[i][2]!=0:
                already_there=circle_in_graphics(G,(V[i][0],V[i][1]),V[i][2])
                if already_there==False:
                    G+=circle((V[i][0],V[i][1]),V[i][2], edgecolor=colord)
                if label!=0:
                    G+=label_circle_by_center_radius((V[i][0], V[i][1]), V[i][2], color='black')
        if label==0:
            G=keep_only_circles(G)
        self._graph=G
        self._super_packing_graph+=G

    def plot_packing(self, label=None, modulus=None, colors=None, min_label_radius=0, min_fontsize=3, figsize=(8, 8), seed_color=None, line_bounds=None, label_fill=0.9, max_fontsize=40):
        """
        Plot the packing using matplotlib, with optional labels and color-by-residue.

        Faster than graph_circles() for large packings because it writes matplotlib
        patches directly.  Circles are drawn largest-first so smaller ones appear on top.
        Font sizes are computed automatically from the packing's coordinate range and
        figsize so labels always scale correctly without manual tuning.

        INPUT:
            label            -- what to draw inside each circle (subject to min_label_radius):
                                  None          no labels (default)
                                  'norm'        cached integer N(curvature), e.g. 7 or -41
                                  'curvature'   the curvature itself in algebraic form;
                                                integers stay integers, rationals display as
                                                p/q, and quadratic elements use latex notation
                                                e.g. '$\\sqrt{2}+3$'.  Every label is rendered
                                                as mathtext, so integer and algebraic
                                                curvatures come out in the same font.
            modulus          -- if set to an integer m, fill each circle with a color
                                determined by N(curvature) mod m (default None = white fill)
            colors           -- dict {residue: color} overriding the built-in pastel palette;
                                only used when modulus is not None
            min_label_radius -- circles with |radius| below this are not labeled (default 0,
                                i.e. every generated circle is labeled). Raise this to thin out
                                labels on very large packings for a cleaner look.
            min_fontsize     -- font size floor in points for labels on very small circles, so
                                a tiny circle still gets a (small) legible label rather than
                                being silently skipped (default 3). A label held up by this
                                floor can still overflow its circle -- that is the point of
                                the floor; raise min_label_radius to drop those labels instead.
            max_fontsize     -- font size ceiling in points for labels on the largest circles
                                (default 40)
            figsize          -- matplotlib figure size in inches (default (8, 8))
            seed_color       -- if set, the 4 starting circles (self._circles[:4]) are outlined
                                in this color (with a thicker edge) instead of the default
                                black, so the seed quadruple stands out from the rest of the
                                packing without changing any fill colors (default None = no
                                highlighting)
            line_bounds      -- (xlo, xhi) x-endpoints for drawing the straight lines of the
                                packing (curvature 0, stored with radius 0 and height V[i][1]).
                                Default None spans the x-extent of all the circles, as
                                graph_circles() does.
            label_fill       -- fraction of the circle's radius a label is allowed to fill
                                (default 0.9). Labels are sized so the whole text box fits
                                inside the circle, so a wide algebraic curvature such as
                                '2*sqrt(2)+3' is set smaller than a one-digit label on a
                                circle of the same size. Lower it for more breathing room,
                                raise it toward 1 for tighter-packed text.

        OUTPUT:
            (fig, ax) -- matplotlib Figure and Axes; display with plt.show() or savefig()

        EXAMPLE::

            sage: import matplotlib.pyplot as plt
            sage: q.bounded_generate_circles(50)
            sage: fig, ax = q.plot_packing()                          # circles only
            sage: fig, ax = q.plot_packing(label='norm', modulus=4)   # colored + norm labels
            sage: fig, ax = q.plot_packing(label='curvature')         # algebraic curvature labels
            sage: fig, ax = q.plot_packing(seed_color='red')          # highlight the starting quadruple
            sage: plt.show()
        """
        import colorsys
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        palette = ['#D3D3D3', '#AED6F1', '#A9DFBF', '#F9E79F', '#F1948A', '#D7BDE2', '#FAD7A0', '#A2D9CE']
        if colors is None and modulus is not None:
            if modulus <= len(palette):
                colors = {i: palette[i] for i in range(modulus)}
            else:
                # palette only has 8 entries; cycling it for larger modulus makes
                # residues 8 apart look identical, so generate one evenly-spaced
                # pastel hue per residue instead
                colors = {i: colorsys.hsv_to_rgb(i/modulus, 0.35, 0.95) for i in range(modulus)}

        V = self._circles
        N = self._norms
        K = self._curvatures

        #x-extent of the actual circles (a line is stored as [0, height, 0], so including
        #the radius-0 entries here would drag the extent back toward x=0)
        circ_x = [(float(V[i][0]) - abs(float(V[i][2])), float(V[i][0]) + abs(float(V[i][2])))
                  for i in range(len(V)) if V[i][2] != 0]
        if circ_x:
            xmin = min(lo for lo, hi in circ_x)
            xmax = max(hi for lo, hi in circ_x)
        else:
            xmin, xmax = -1.0, 1.0
        if line_bounds is not None:
            xmin, xmax = float(line_bounds[0]), float(line_bounds[1])

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect('equal')
        ax.axis('off')

        order = sorted(range(len(V)), key=lambda i: -abs(float(V[i][2])) if V[i][2] != 0 else float('inf'))

        to_label = []   # (label string, cx, cy, radius); sized once the axes scale is known

        for i in order:
            try:
                cx, cy, r = float(V[i][0]), float(V[i][1]), float(V[i][2])
            except Exception:
                continue
            if seed_color is not None and i < 4:
                ec, lw = seed_color, 2.5
            else:
                ec, lw = 'black', 0.5

            #radius 0 flags a straight line (curvature 0); it is horizontal at height cy,
            #drawn across the x-extent of the packing exactly like graph_circles() does
            if r == 0:
                ax.plot([xmin, xmax], [cy, cy], color=ec, linewidth=lw, solid_capstyle='butt')
                continue
            radius = abs(r)

            if modulus is not None and i < len(N):
                fc = colors.get(int(N[i]) % modulus, 'white')
            else:
                fc = 'white'

            ax.add_patch(mpatches.Circle((cx, cy), radius, facecolor=fc, edgecolor=ec, linewidth=lw))

            #r<0 flags the enclosing (negative-curvature) circle -- it's always the
            #largest circle in the packing, so it would otherwise always get labeled
            if label is not None and radius >= min_label_radius and r > 0:
                if label == 'norm' and i < len(N):
                    #same formatter as the curvature labels: integers (the usual case)
                    #print as integers, and a symbolic norm gets latexed instead of
                    #showing raw python like '12*sqrt(2) + 17'
                    lstr = _format_curvature(N[i])
                elif label == 'curvature' and i < len(K):
                    lstr = _format_curvature(K[i])
                else:
                    lstr = None

                if lstr:
                    to_label.append((lstr, cx, cy, radius))

        ax.autoscale_view()

        if to_label:
            #Points per coordinate unit, read off the finished axes rather than
            #assumed from figsize: the axes box is smaller than the figure, and with
            #an equal aspect ratio it is whichever of width/height binds that sets the
            #scale. Deriving it from figsize/span alone overestimated it by ~30%, which
            #is what pushed long labels outside their circles.
            ax.apply_aspect()
            p0 = ax.transData.transform((0.0, 0.0))
            p1 = ax.transData.transform((1.0, 0.0))
            font_scale = abs(p1[0] - p0[0]) * 72.0 / fig.dpi   # pts per coordinate unit

            for lstr, cx, cy, radius in to_label:
                #fit the label's whole box inside the circle: a text box w x h (in
                #points) fits in a circle of radius R when its half-diagonal
                #sqrt((w/2)^2 + (h/2)^2) <= R, so scale the font by
                #   2*fill*R / sqrt(w_em^2 + h_em^2)
                #with fill leaving a margin between the text and the arc. Sizing on
                #character count alone let wide algebraic curvatures such as
                #'2*sqrt(2)+3' run outside their circle.
                w_em, h_em = _label_box_em(lstr)
                #va='center' centres the layout box (which reserves room for
                #descenders) rather than the ink, so the ink sits slightly high;
                #allow for that shift when fitting
                h_em = h_em + 0.25
                r_pts = font_scale * radius        # circle radius in points
                fontsize = 2.0 * label_fill * r_pts / ((w_em**2 + h_em**2)**0.5)
                fontsize = max(min_fontsize, min(max_fontsize, float(fontsize)))
                ax.text(cx, cy, lstr, fontsize=fontsize, ha='center', va='center', zorder=5)

        return fig, ax

    #put the starting 4 circles into a graph
    def _graph_circle(self, color='black', xLbound=-1, xUbound=1):
        """
        Build a fresh Graphics object from the current self._circles list.

        Used internally during initialization and after reductions to reset the base graphic.

        INPUT:
            color   -- edge color for circles (default 'black')
            xLbound -- left x-endpoint for lines (default -1)
            xUbound -- right x-endpoint for lines (default 1)

        OUTPUT:
            A new Sage Graphics object containing the 4 initial circles / lines.
        """
        V=self._circles
        xmin=min([min(V[i][0]+V[i][2],V[i][0]-V[i][2]) for i in range(len(V))])
        xmax=max([max(V[i][0]+V[i][2],V[i][0]-V[i][2])for i in range(len(V))])
        G=Graphics()
        for i in range(0,len(V)):
            if V[i][2]==0:
                G+=line([(xmin,V[i][1]), (xmax, V[i][1])], color=color)
            elif V[i][2]!=0:
                G+=circle((V[i][0],V[i][1]),V[i][2], edgecolor=color)
                #G+=label_circle_by_center_radius((V[i][0], V[i][1]), V[i][2], color='black')
        return G;




######################################################## 
#GENERATE PACKING UP TO CERTAIN CURVATURE OR WORD LENGTH
########################################################




    #outputs the swapped curv and center size compared to the bound
    #1 is swapped curv center is in bound, and curv is in bound
    # -1 is not above
    def _swap_center_size(self, index, lbound,ubound, xmax, ymax, xmin, ymin, grp=4):
        """
        Check whether the circle produced by swapping index lies within bounds (full version).

        This is the full version used by _generate_circles(); it supports both the standard
        Apollonian group (grp=4) and the super-Apollonian group (grp=8), and handles
        zero-curvature (line) circles and negative curvatures (super packing).

        INPUT:
            index  -- integer; the generator index (0--3 for grp=4, 0--7 for grp=8)
            lbound -- lower curvature bound
            ubound -- upper curvature bound
            xmax   -- maximum x-coordinate (curv-weighted)
            ymax   -- maximum y-coordinate (curv-weighted)
            xmin   -- minimum x-coordinate (default 0)
            ymin   -- minimum y-coordinate (default 0)
            grp    -- 4 for standard Apollonian, 8 for super-Apollonian (default 4)

        OUTPUT:
             1 if the swapped circle satisfies all bounds
            -1 otherwise

        NOTATION:
            For k != 0: xval = k*x, yval = k*y (entries from the Descartes matrix).
            For k  = 0: uses co-curvature for height and midpoint of x-range.
            For k  < 0: uses |k| (relevant for super-packing outer circles).
        """
        if grp==4:
            swapD, swapDM=self.swap(index)
            curv=swapD[index%4]
            if curv!=0:
                xval= swapDM[1][index%4]/swapD[index%4]
                yval= swapDM[2][index%4]/swapD[index%4]
            if curv==0: #only really want to check how high up it goes and basically just for strip packing
                yval=swapDM[3][index%4]
                xval=(xmax-xmin)/2
                curv=(ubound-lbound)/2
            if curv<0: #mainly for superpacking things
                curv=curv.abs()
            if lbound<=curv<=ubound and xmin <= xval<=xmax and ymin<=yval<=ymax:
                return 1
            else:
                return -1
        else:
            #we're doing a super swap 
            swapD, swapDM=self.super_swap(index)
            center_size_4=[]
            for i in range(4):
                curv=swapD[i]
                if curv!=0:
                    xval= swapDM[1][i%4]/swapD[i%4]
                    yval= swapDM[2][i%4]/swapD[i%4]
                if curv==0: #only really want to check how high up it goes and basically just for strip packing
                    yval=swapDM[3][i%4]
                    xval=(xmax-xmin)/2
                    curv=(ubound-lbound)/2
                if curv<0: #mainly for superpacking things
                    curv=curv.abs()
                if lbound<=curv<=ubound and xmin <= xval<=xmax and ymin<=yval<=ymax:
                    center_size_4.append(1)
                    return 1
                else:
                    center_size_4.append(-1)
                    return -1



    #generate circles based on a given method for unbounded or bounded
    def _generate_circles(self, mode, method_name=None, *args, **kwargs):
        """
        Core DFS traversal of the Apollonian (or super-Apollonian) graph to generate circles.

        Walks the group-action tree, applying swaps and checking a stopping criterion at
        each node.  When a node passes the criterion, the new circle is added and the search
        continues deeper; otherwise it backtracks.

        INPUT:
            mode        -- 'word' to stop at a fixed word length; any other string to use a
                           curvature / spatial bound method.
            method_name -- name (string) of the bound-checking method to call, e.g.
                           '_swap_size' or '_swap_center_size'.  Unused when mode='word'.
            *args       -- positional arguments forwarded to the bound-checking method.
            **kwargs    -- keyword arguments; must include:
                           grp        -- 4 (Apollonian) or 8 (super-Apollonian)
                           wordLength -- maximum word length when mode='word'

        Modifies self._circles (or self._supercircles) in place via _swap_add().
        Restores self._entries and self._desc_matrix to their original values on exit.
        """
        og_quad=self._entries
        og_descM=self._desc_matrix

        quad2=self._entries #just focus on the quadruple

        counterQuad=0 #counts the number of swaps done
        changingQuad=[]
        prevSwap=-1 #we start with no swaps
        swap=0
        location=[-1]

        #using the super apollonian group or usual apollonian grp
        grp=kwargs.get("grp")
        wordLength=kwargs.get("wordLength")

        while swap!=grp:
            #print("beginning", self, quad2, changingQuad, location, prevSwap, swap)
            swap=swap%grp
            if swap==prevSwap: #dont undo the previous swap
                swap=(swap+1)%grp
            #defaut mode is the curvature model
            if mode=="word": #if we want to walk the graph using word length
                result=-1
                if len(location)-1<wordLength:
                    result=1
            else: #if we want to traverse the graph based on curvatures
                method=getattr(self, method_name)
                result=method(swap,*args,**kwargs)

            if result!=-1: #this just decides if we continue walking or not
                self._swap_add(swap,grp) #new circle fits bound so add it to the generated circles
                location.append(swap) #keep track of location
                prevSwap=swap #keep track of previous swap
                changingQuad=self._entries
                swap=-1 #reset swap position to 0 after adding 1
                counterQuad=-1 #reset the number of swaps weve done after adding 1
            swap=(swap+1)
            counterQuad+=1
            #print("middle", self, quad2, changingQuad, location, prevSwap, swap)
            while counterQuad>=grp-1 and prevSwap!=-1: #path terminated, need to check for more paths
                if location[len(location)-1]==grp-1:
                    self.swap_change(location[len(location)-1],grp)
                    changingQuad=self._entries #go back one step to continue searching
                    counterQuad=self._prev_count(location) #update counter, goes back two if at the end
                    swap=location[len(location)-1]+1 #next swap
                    location=location[:-1] #update location
                    prevSwap=location[len(location)-1] #previous swap
                if location[len(location)-1]!=grp-1 and prevSwap!=-1:
                    self.swap_change(location[len(location)-1],grp) #change back to previous quad w/o adding circles
                    changingQuad=self._entries #go back one step to continue searching
                    swap=location[len(location)-1]+1 #next swap
                    counterQuad=self._prev_count(location) #update counter
                    location=location[:-1] #update location
                    prevSwap=location[len(location)-1] #previous swap
            #print("end", self, quad2, changingQuad, location, prevSwap, swap)
        self._entries=og_quad
        self._desc_matrix=og_descM

    #finding number of checks done, only used in _bounded_generate_circles
    #finds the number of quadruples checked in the previous step of algorithm
    def _prev_count(self,loc):
        """
        Compute the number of swap checks performed in the previous DFS step.

        Used by _generate_circles() when backtracking to correctly reset the counterQuad
        variable so the traversal knows how many sibling branches have already been explored.

        INPUT:
            loc -- current location list (list of swap indices from root to current node)

        OUTPUT:
            Integer count of swaps already tried at the parent node, excluding the swap
            that led to the current node.
        """
        count=0
        last=len(loc)-1 #index of where currently
        swapCount=loc[last]+1 #next swap to do
        prev=loc[last-1]
        for i in range(0,swapCount):
            if i!=prev:
                count+=1
        return count

    #generates all the circles in the packing with curvature<bound
    #only for bounded packings
    #adds them to the circles generated
    def bounded_generate_circles(self, bound):
        """
        Generate all circles in the Apollonian packing with curvature < bound.

        Traverses the Apollonian group graph via DFS, adding every circle whose
        curvature is less than bound to self._circles.  For bounded (finite) packings only
        (all four initial curvatures positive).

        INPUT:
            bound -- numeric upper bound on curvature (exclusive)

        Modifies self._circles in place.

        EXAMPLE::

            sage: q = DescartesQuad([-1, 2, 2, 3])
            sage: q.bounded_generate_circles(50)
            sage: q.graph_circles()
            sage: q._graph.show()
        """
        self._generate_circles("curv","_swap_size", bound, grp=4)

    #generates all the circles in the packing with lbound<=curvature<=ubound and center in bound
    #adds them to the circles generated
    def unbounded_generate_circles(self, lbound, ubound, xmax, ymax, xmin=0,  ymin=0):
        """
        Generate circles in the packing whose curvature and center lie within given bounds.

        For strip packings or other unbounded configurations where curvature alone is not
        a finite stopping criterion.

        INPUT:
            lbound -- lower bound on curvature (inclusive)
            ubound -- upper bound on curvature (inclusive)
            xmax   -- maximum x-coordinate of circle center
            ymax   -- maximum y-coordinate of circle center
            xmin   -- minimum x-coordinate of circle center (default 0)
            ymin   -- minimum y-coordinate of circle center (default 0)

        Modifies self._circles in place.

        EXAMPLE::

            sage: q = DescartesQuad([0, 1, 1, 4])
            sage: q.unbounded_generate_circles(0, 20, 5, 3)
        """
        self._generate_circles("curv","_swap_center_size", lbound, ubound, xmax, ymax, xmin,  ymin,grp=4)


    #generate the packing while only going a wordLength in
    #adds them to circles generated
    #main purpose is for strip packings
    def word_generate_circles(self, wordLength):
        """
        Generate circles by walking the Apollonian graph up to a fixed word length.

        Adds every circle reachable by a reduced word of length <= wordLength in the
        Apollonian group generators {S_0, S_1, S_2, S_3}.  Useful for strip packings
        or when a curvature bound is not practical.

        INPUT:
            wordLength -- non-negative integer; maximum depth of the DFS traversal

        Modifies self._circles in place.

        EXAMPLE::

            sage: q = DescartesQuad([0, 0, 1, 1])
            sage: q.word_generate_circles(6)
        """
        self._generate_circles("word", wordLength=wordLength,grp=4)


    
    
##################################################
#SUPER PACKING
###########################################################

    #generate superpacking using a word length
    def super_packing_word(self,wordLength):
        """
        Generate the super-Apollonian packing by walking the super group up to wordLength.

        The super-Apollonian group has 8 generators (indices 0--7): the 4 standard
        Apollonian generators and their 4 transposes.  This produces the dual / super packing.

        INPUT:
            wordLength -- non-negative integer; maximum depth of the DFS traversal

        Modifies self._supercircles in place.

        EXAMPLE::

            sage: q = DescartesQuad([-1, 2, 2, 3])
            sage: q.super_packing_word(4)
            sage: q.super_graph_circles()
            sage: q._super_packing_graph.show()
        """
        self._generate_circles("word",wordLength=wordLength, grp=8)

    #generate superpacking using bounds on curv and center
    #this isn't working rn
    def super_packing_bound(self,lbound, ubound, xmax, ymax, xmin,  ymin):
        """
        Generate the super-Apollonian packing with curvature and spatial bounds.

        Note: this method is currently not working correctly.

        INPUT:
            lbound -- lower curvature bound
            ubound -- upper curvature bound
            xmax   -- maximum x-coordinate of circle center
            ymax   -- maximum y-coordinate of circle center
            xmin   -- minimum x-coordinate of circle center
            ymin   -- minimum y-coordinate of circle center

        Modifies self._supercircles in place.
        """
        self._generate_circles("curv","_swap_center_size", lbound, ubound, xmax, ymax, xmin,  ymin,grp=8)

    def super_graph_circles(self, color='green',xLbound=-1, xUbound=1):
        """
        Render all super-packing circles in self._supercircles into self._super_packing_graph.

        Circles already present in the Graphics object are skipped (no duplicates).
        Lines (radius 0) are drawn as horizontal segments between xLbound and xUbound.

        INPUT:
            color   -- edge color for circles (default 'green')
            xLbound -- left x-endpoint for lines (default -1)
            xUbound -- right x-endpoint for lines (default 1)

        Modifies self._super_packing_graph in place.

        EXAMPLE::

            sage: q.super_packing_word(4)
            sage: q.super_graph_circles()
            sage: q._super_packing_graph.show()
        """
        V=self._supercircles
        G=self._super_packing_graph
        for i in range(0,len(V)):
            if V[i][2]==0:
                G+=line([(xLbound,V[i][1]), (xUbound, V[i][1])])
            elif V[i][2]!=0:
                already_there=circle_in_graphics(G,(V[i][0],V[i][1]),V[i][2])
                if already_there==False:
                    G+=circle((V[i][0],V[i][1]),V[i][2], edgecolor=color)
        self._super_packing_graph=G
