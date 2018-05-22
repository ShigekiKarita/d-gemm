import std.stdio;
import std.exception;

import mir.ndslice;
import mir.blas : gemm;
import G = glas.ndslice;

auto naiveGemm(S, T=DeepElementType!S)(const T alpha, const S a, const S b, const T beta, ref S c) if (isSlice!S)
in {
    assert(c.length!0 == a.length!0);
    assert(a.length!1 == b.length!0);
    assert(c.length!1 == b.length!1);
} do {
    alias E = DeepElementType!S;
    foreach (i; 0 .. a.length!0) {
        foreach (j; 0 .. b.length!1) {
            E ab = 0;
            foreach (k; 0 .. a.length!1) {
                ab += a[i, k] * b[k, j];
            }
            c[i, j] = alpha * ab + beta * c[i, j];
        }
    }
}

auto asarray(S)(S s) if (isSlice!S && isContiguousVector!S) {
    return s._iterator[0 .. s.length!0];
}

import core.simd;

pragma(LDC_intrinsic, "llvm.x86.fma.vfmadd.pd.256")
double4 fmadd(double4, double4, double4);

double dotAvx(const double[] vec1, const double[] vec2) {
    double4 u1 = [0, 0, 0, 0];
    double4 w1, x1;
    immutable n = vec1.length;
    immutable m = n - (n % 4);
    for (size_t i = 0; i < m; i += 4) {
        w1.array = vec1[i..i+4];
        x1.array = vec2[i..i+4];
        // u1 = fmadd(w1, x1, u1);
        u1 += w1 * x1;
    }

    double ret = u1.array[0] + u1.array[1] + u1.array[2] + u1.array[3];
    foreach (i; m .. n) {
        ret += vec1[i] * vec2[i];
    }
    return ret;
}

unittest {
    double[] a = [1, 2, 3, 4, 1, 2, 3, 4, 5];
    double[] b = [-1, 2, -3, 4, -1, 2, -3, 4, 5];
    // writeln(dotAvx(a, b), " vs ", (1*-1 + 2*2 + 3*-3 + 4*4) * 2);
    assert(dotAvx(a, b) == (1*-1 + 2*2 + 3*-3 + 4*4) * 2 + 25);
}


pragma(inline, true)
nothrow @nogc
auto mapDot(S)(in S a, in S b) if (isSlice!S) {
    import mir.math : sum;
    return sum!"fast"(a[] * b[]);
}

auto mapGemm(S, T=DeepElementType!S)(const T alpha, const S a, const S b, const T beta, ref S c) if (isSlice!S)
in {
    assert(c.length!0 == a.length!0);
    assert(a.length!1 == b.length!0);
    assert(c.length!1 == b.length!1);
} do {
    import std.numeric : dotProduct;
    alias E = DeepElementType!S;
    // TODO check contiguous or not like lubeck
    auto bt = b.transposed.slice;
    foreach (i; 0 .. a.length!0) {
        foreach (j; 0 .. b.length!1) {
            c[i, j] = alpha * mapDot(a[i], bt[j]) + beta * c[i, j];
        }
    }
}

auto dotGemm(S, T=DeepElementType!S)(const T alpha, const S a, const S b, const T beta, ref S c) if (isSlice!S)
in {
    assert(c.length!0 == a.length!0);
    assert(a.length!1 == b.length!0);
    assert(c.length!1 == b.length!1);
} do {
    import std.numeric : dotProduct;
    alias E = DeepElementType!S;
    // TODO check contiguous or not like lubeck
    auto bt = b.transposed.slice;
    foreach (i; 0 .. a.length!0) {
        foreach (j; 0 .. b.length!1) {
            E ab = dotAvx(a[i].asarray, bt[j].asarray);
            c[i, j] = alpha * ab + beta * c[i, j];
        }
    }
}


auto dotParallelGemm(S, T=DeepElementType!S)(
    const T alpha, const S a, const S b, const T beta, ref S c) if (isSlice!S)
in {
    assert(c.length!0 == a.length!0);
    assert(a.length!1 == b.length!0);
    assert(c.length!1 == b.length!1);
} do {
    import std.numeric : dotProduct;
    import std.parallelism;
    alias E = DeepElementType!S;
    // TODO check contiguous or not like lubeck
    auto bt = b.transposed.slice;
    foreach (i; a.length!0.iota.parallel) {
        foreach (j; b.length!1.iota.parallel) {
            c[i, j] = alpha * mapDot(a[i], bt[j]) + beta * c[i, j];
        }
    }
}


auto dotParallelUnrollGemm(size_t unroll=16, S, T=DeepElementType!S)(
    const T alpha, const S a, const S b, const T beta, ref S c) if (isSlice!S)
in {
    assert(c.length!0 == a.length!0);
    assert(a.length!1 == b.length!0);
    assert(c.length!1 == b.length!1);
} do {
    import std.numeric : dotProduct;
    import std.parallelism;
    alias E = DeepElementType!S;
    // TODO check contiguous or not like lubeck
    auto bt = b.transposed.slice;
    foreach (i; a.length!0.iota.parallel) {
        immutable bsteps = b.length!1 / unroll;
        immutable bremain = b.length!1 % unroll;
        foreach (bs; bsteps.iota.parallel) {
            auto c_ = c[i, bs * unroll .. (bs + 1) * unroll + 1];
            const b_ = bt[bs * unroll .. $, 0 .. $];
            static foreach (j; 0 .. unroll) {
                c_[j] = alpha * mapDot(a[i], b_[j]) + beta * c_[j];
            }
        }
        foreach (j; b.length!1 - bremain .. b.length!1) {
            c[i, j] = alpha * mapDot(a[i], bt[j]) + beta * c[i, j];
        }
    }
}


import ldc.attributes;
import ldc.intrinsics;

@fastmath
auto expectDotParallelUnrollGemm(size_t unroll=16, S, T=DeepElementType!S)(
    const T alpha, const S a, const S b, const T beta, ref S c) if (isSlice!S)
in {
    assert(c.length!0 == a.length!0);
    assert(a.length!1 == b.length!0);
    assert(c.length!1 == b.length!1);
} do {
    import std.parallelism;
    alias E = DeepElementType!S;
    // TODO check contiguous or not like lubeck
    auto bt = b.transposed.slice;
    foreach (i; a.length!0.iota.parallel) {
        immutable bsteps = b.length!1 / unroll;
        immutable bremain = b.length!1 % unroll;
        foreach (bs; bsteps.iota.parallel) {
            auto c_ = c[i, bs * unroll .. (bs + 1) * unroll + 1];
            const b_ = bt[bs * unroll .. $, 0 .. $];
            static foreach (j; 0 .. unroll) {
                if (llvm_expect(alpha == 1.0 && beta == 0.0, true)) {
                    c_[j] = mapDot(a[i], b_[j]);
                } else {
                    c_[j] = alpha * mapDot(a[i], b_[j]) + beta * c_[j];
                }
            }
        }
        foreach (j; b.length!1 - bremain .. b.length!1) {
            c[i, j] = alpha * mapDot(a[i], bt[j]) + beta * c[i, j];
        }
    }
}

alias glasGemm = G.gemm;
alias blasGemm = gemm;

void main()
{
    import std.format;
    import std.datetime.stopwatch;

    import numir : uniform, approxEqual, zeros;
    enum nTrial = 10;
    enum names = [
        "blas",
        // "naive",
        "map",
        "dot",
        "dotParallel",
        "dotParallelUnroll",
        "expectDotParallelUnroll",
        "glas"
        ];
    foreach (m; [128, 256, 512, 1024]) {
        writefln("m = %d", m);
        auto a = uniform(m, m).slice;
        auto b = uniform(m, m).slice;
        auto au = a.universal;
        auto bu = b.universal;
        static foreach (name; names) {
            static if (name == "glas") {
                mixin("auto c" ~ name ~ " = zeros(a.length!0, b.length!1).universal;");
            } else {
                mixin("auto c" ~ name ~ " = zeros(a.length!0, b.length!1);");
            }

            {
                static if (name == "glas") {
                    mixin(format!("auto result = benchmark!(() { %sGemm(1.0, a.universal, b.universal, 0.0, c%s); })(%d);")(name, name, nTrial));
                } else {
                    mixin(format!("auto result = benchmark!(() { %sGemm(1.0, a, b, 0.0, c%s); })(%d);")(name, name, nTrial));
                }
                writeln(name, ":", result[0] / nTrial);
                mixin("enforce(approxEqual(cblas, c" ~ name ~ "), \"" ~ name ~ " is invalid\");");
            }
        }
    }
}
