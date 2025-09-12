from sympy import Function, symbols, simplify, factor, pprint, latex, cse, collect
from sympy.abc import alpha


g_prime = Function("g'")

class g(Function):
    def fdiff(self, argindex=1):
        return g_prime(self.args[0])
    

class l(Function):
    def fdiff(self, argindex=1):
        return g(self.args[0])


def main():
    variables = {}

    (n0, n1, n2, n3, n4) = symbols("n:5")
    (s0, s1, s2, s3, s4) = symbols("s:5")
    (w0, w1, w2, w3, w4) = symbols('w:5')

    t1 = w0 - alpha * (g(w0) + n0 * s0)
    t2 = t1 - alpha * (g(t1) + n1 * s1)
    t3 = t2 - alpha * (g(t2) + n2 * s2)
    t4 = t3 - alpha * (g(t3) + n3 * s3)
    t5 = t4 - alpha * (g(t4) + n4 * s4)
    for s in [s4, s3, s2, s1, s0]:
        print(r"\frac{d(t5)}{d(" + str(s) + r")} &= ", end="")
        print(latex(factor(l(t5).diff(s)
                .subs(w0 - alpha * (g(w0) + n0 * s0), w1)
                .subs(w1 - alpha * (g(w1) + n1 * s1), w2)
                .subs(w2 - alpha * (g(w2) + n2 * s2), w3)
                .subs(w3 - alpha * (g(w3) + n3 * s3), w4)
                .subs(w4 - alpha * (g(w4) + n4 * s4), symbols('w5'))
                )), end=r" \\" + "\n")
    
    exit()

    for key in filter(lambda key: key.startswith('w'), variables):
        eqn = variables[key]
        weight_num = int(key.split("_")[-1])
        if weight_num == 3:
            for sigma_num in range(weight_num-1, -1, -1):
                sigma = variables[f"s_{sigma_num}"]
                print(latex(factor(eqn)))
                print(f"d({key}) / d(s_{sigma_num})")
                print(latex(simplify(eqn.diff(sigma))))
                print()
            
            print("\n---\n")


if __name__ == "__main__":
    main()