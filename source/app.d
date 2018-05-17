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
            E ab = dotProduct(a[i].asarray, bt[j].asarray);
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
            E ab = dotProduct(a[i].asarray, bt[j].asarray);
            c[i, j] = alpha * ab + beta * c[i, j];
        }
    }
}


auto dotParallelUnrollGemm(size_t unroll, S, T=DeepElementType!S)(
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
    // immutable asteps = a.length!0 / unroll;
    // immutable aremain = a.length!0 % unroll;
    foreach (i; a.length!0.iota.parallel) {
        // const a_ = a[as * unroll .. $, 0 .. $];
        // size_t i;
        // static foreach (ai; 0 .. unroll) {
        //     i = as * unroll + ai;
        immutable bsteps = b.length!1 / unroll;
        immutable bremain = b.length!1 % unroll;
        foreach (bs; bsteps.iota.parallel) {
            E ab;
            auto c_ = c[i, bs * unroll .. (bs + 1) * unroll + 1];
            const b_ = bt[bs * unroll .. $, 0 .. $];
            static foreach (j; 0 .. unroll) {
                ab = dotProduct(a[i].asarray, b_[j].asarray);
                c_[j] = alpha * ab + beta * c_[j];
            }
        }
        foreach (j; b.length!1 - bremain .. b.length!1) {
            E ab = dotProduct(a[i, 0..$], b[0..$, j]);
            c[i, j] = alpha * ab + beta * c[i, j];
        }
    }
}



void main()
{
    import std.datetime.stopwatch;
    foreach (m; [128, 256, 512, 1024]) {
        auto a = iota(m, m).as!double.slice;
        auto b = iota(m, m).as!double.slice;
        auto au = a.universal;
        auto bu = b.universal;

        auto cNaive = slice!double(a.length!0, b.length!1);
        auto cDot = slice!double(a.length!0, b.length!1);
        auto cParallel = slice!double(a.length!0, b.length!1);
        auto cParallelUnroll = slice!double(a.length!0, b.length!1);
        auto cGlas = slice!double(a.length!0, b.length!1).universal;
        auto cBlas = slice!double(a.length!0, b.length!1);

        auto results = benchmark!(
            () { naiveGemm(1, a, b, 0, cNaive); },
            () { dotGemm(1, a, b, 0, cDot); },
            () { dotParallelGemm(1, a, b, 0, cParallel); },
            () { dotParallelUnrollGemm!16(1, a, b, 0, cParallelUnroll); },
            () { G.gemm(1.0, au, bu, 0.0, cGlas); },
            () { gemm(1, a, b, 0, cBlas); }
            )(10);
        writefln!"==== benchmark (m=%d) ===="(m);
        writeln("naive:", results[0]);
        writeln("std.numeric.dotProduct:", results[1]);
        writeln("std.parallelism.parallel:", results[2]);
        writeln("unroll:", results[3]);
        writeln("glas:", results[4]);
        writeln("blas:", results[5]);
        // enforce(approxEqual(cBlas, cNaive), "blas is invalid");
        // enforce(approxEqual(cDot, cNaive), "dot is invalid");
        // enforce(approxEqual(cParallel, cNaive), "parallel is invalid");
        // enforce(approxEqual(cGlas, cNaive), "glas is invalid");
        // enforce(approxEqual(cParallelUnroll, cNaive), "unroll is invalid");
    }
}
