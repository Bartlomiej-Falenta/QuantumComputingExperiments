import numpy as np
import matplotlib.pyplot as plt
import argparse


plt.rcParams['text.usetex'] = True

# ============================================================
# Envelope & base infrastructure
# ============================================================

class Envelope:
    name = "base"

    def __call__(self, x, **kwargs):
        raise NotImplementedError

    def latex(self, **kwargs):
        raise NotImplementedError


ENVELOPES = {}

def register(envelope):
    ENVELOPES[envelope.name] = envelope

BASES = {
    "sin": ("sin",),
    "cos": ("cos",),
    #"trig": ("sin", "cos"),
    "poly": ("poly",),
    #"mixed": ("sin", "cos", "poly"),
}


# ============================================================
# Envelope implementations (ALL are L^2)
# ============================================================

class GaussianEnvelope(Envelope):
    name = "gaussian"

    def __call__(self, x, sigma=1.0):
        return np.exp(-(x**2) / sigma**2)

    def latex(self, sigma=1.0):
        return rf"e^{{-x^2/{sigma}^2}}"


class SuperGaussianEnvelope(Envelope):
    name = "super_gaussian"

    def __call__(self, x, sigma=1.0, p=4):
        return np.exp(-(np.abs(x)/sigma)**p)

    def latex(self, sigma=1.0, p=4):
        return rf"e^{{-(|x|/{sigma})^{p}}}"


class LaplaceEnvelope(Envelope):
    name = "laplace"

    def __call__(self, x, sigma=1.0):
        return np.exp(-np.abs(x)/sigma)

    def latex(self, sigma=1.0):
        return rf"e^{{-|x|/{sigma}}}"


class RationalEnvelope(Envelope):
    name = "rational"

    def __call__(self, x, sigma=1.0, p=2.0):
        return 1 / (1 + (np.abs(x)/sigma)**p)

    def latex(self, sigma=1.0, p=2.0):
        return rf"\frac{{1}}{{1+(|x|/{sigma})^{p}}}"


class SmoothBumpEnvelope(Envelope):
    name = "smooth_bump"

    def __call__(self, x, L=2.0):
        y = np.zeros_like(x)
        mask = np.abs(x) < L
        y[mask] = np.exp(-1/(1-(x[mask]/L)**2))
        return y

    def latex(self, L=2.0):
        return rf"e^{{-1/(1-(x/{L})^2)}}\,\mathbf{{1}}_{{|x|<{L}}}"


# Register all envelopes
register(GaussianEnvelope())
register(SuperGaussianEnvelope())
register(LaplaceEnvelope())
register(RationalEnvelope())
register(SmoothBumpEnvelope())

# ============================================================
# Function generator
# ============================================================
def generate_L2_function(
    N=7,
    envelope_name=None,
    envelope_params=None,
    amp_scale=1.0,
    freq_scale=1.0,
    basis=None,
    poly_degree=2,
    seed=None
):
    rng = np.random.default_rng(seed)

    # -----------------------------
    # Envelope selection
    # -----------------------------
    if envelope_name is None:
        envelope = rng.choice(list(ENVELOPES.values()))
        if envelope_params is None:
            envelope_params = {}
    else:
        envelope = ENVELOPES[envelope_name]
        if envelope_params is None:
            envelope_params = {}

    # -----------------------------
    # Basis setup
    # -----------------------------
    if basis is None:
        basis_seq = BASES["sin"]
    else:
        basis_seq = BASES[basis]

    # -----------------------------
    # Oscillatory core
    # -----------------------------
    a = rng.normal(scale=amp_scale, size=N)
    w = rng.uniform(0.5, 3.0, size=N) * freq_scale
    p = rng.uniform(0, 2*np.pi, size=N)
    b = rng.choice(basis_seq, size=N)

    # Make sure poly degrees is array of length N
    if "poly" in basis_seq:
        poly_degrees = np.full(N, poly_degree)
    else:
        poly_degrees = np.zeros(N, dtype=int)  # will not be used

    # -----------------------------
    # Rational envelope check (L2 safe)
    # -----------------------------
    if "poly" in basis_seq and envelope.name == "rational":
        env_p = envelope_params.get("p", 2.0)
        if env_p <= poly_degree + 0.5:
            raise ValueError(
                f"Rational envelope requires p > poly_degree + 1/2 "
                f"(got p={env_p}, degree={poly_degree})"
            )

    # -----------------------------
    # Function
    # -----------------------------
    def f(x):
        s = np.zeros_like(x)
        for ak, wk, pk, bk, pd in zip(a, w, p, b, poly_degrees):
            if bk == "sin":
                s += ak * np.sin(wk*x + pk)
            elif bk == "cos":
                s += ak * np.cos(wk*x + pk)
            elif bk == "poly":
                s += ak * (wk*x + pk)**pd
        return s * envelope(x, **envelope_params)

    return f, (a, w, p, b, poly_degrees), envelope, envelope_params, basis


# ============================================================
# LaTeX generation
# ============================================================
def latex_formula(core_params, envelope, envelope_params):
    a, w, p, b, poly_degrees = core_params
    terms = []

    for ak, wk, pk, bk, pd in zip(a, w, p, b, poly_degrees):
        if bk == "sin":
            func = r"\sin"
            term = rf"{ak:.1f}{func}({wk:.1f}x + {pk:.1f})"
        elif bk == "cos":
            func = r"\cos"
            term = rf"{ak:.1f}{func}({wk:.1f}x + {pk:.1f})"
        elif bk == "poly":
            # pd must be integer for MathText, ensure it
            term = rf"{ak:.1f}({wk:.1f}x + {pk:.1f})^{{{int(pd)}}}"

        # This one needs to be corrected...
        for i in range(len(terms)):
            terms.append(term)
            if i % 3 == 0:
                terms.append(r"\\")
        
    core = " + ".join(terms)

    # envelope latex: remove double backslash, use single
    env_latex = envelope.latex(**envelope_params)
    env_latex = env_latex.replace("\\\\", "\\")

    formula = r"f(x) = \left(" + core + env_latex + r"\right) \\,"

    return formula


# ============================================================
# Demo / test run
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate semi-random L2 functions"
    )

    parser.add_argument("--envelope",choices=ENVELOPES.keys(), help="Envelope type")
    parser.add_argument("--sigma", type=float, help="Scale parameter")
    parser.add_argument("--p", type=float, help="Exponent parameter")
    parser.add_argument("--L", type=float, help="Compact support radius")
    parser.add_argument("--N", type=int, default=7, help="Number of oscillatory components")
    parser.add_argument("--basis", choices=BASES.keys(), help="Basis for oscillatory core")
    parser.add_argument("--poly_degree", type=int, default=2, help="Polynomial degree for poly basis (>= 0)")
    parser.add_argument("--seed", type=int, help="Random seed")

    args = parser.parse_args()

    # --------------------------------------------
    # Collect envelope parameters explicitly
    # --------------------------------------------

    envelope_params = {}

    if args.sigma is not None:
        envelope_params["sigma"] = args.sigma
    if args.p is not None:
        envelope_params["p"] = args.p
    if args.L is not None:
        envelope_params["L"] = args.L

    # --------------------------------------------
    # Generate function
    # --------------------------------------------

    f, core_params, env, env_params, basis = generate_L2_function(
        N=args.N,
        envelope_name=args.envelope,
        envelope_params=envelope_params if args.envelope else None,
        basis=args.basis,
        poly_degree=args.poly_degree,
        seed=args.seed
    )

    x = np.linspace(-8, 8, 3000)
    y = f(x)

    sin = "sin"     #dumbass correction, who cares
    plt.figure(figsize=(16, 9))
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"L² function with envelope: {env.name} and basis: {args.basis if args.basis is not None else sin}")

    formula = latex_formula(core_params, env, env_params)

    font_1 = {
        'fontsize': '10',
        'fontname': 'Nimbus Roman', #ADD: MathJax_Main
        'color': "#108824"
    }

    plt.text(x=0.5, y=-0.29, s=f"${formula}$", fontsize=11, color = "#108824")

    plt.tight_layout()
    plt.show()