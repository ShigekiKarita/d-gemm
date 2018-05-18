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

auto mapDot(S)(const S a, const S b) if (isSlice!S) {
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
            c[i, j] = alpha * mapDot(a[i], bt[j]) + beta * c[i, j];
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



void main()
{
    import std.datetime.stopwatch;
    import numir : uniform, approxEqual, zeros;
    foreach (m; [8, 32, 128, 256, 512, 1024]) {
        auto a = uniform(m, m).slice;
        auto b = uniform(m, m).slice;
        auto au = a.universal;
        auto bu = b.universal;

        auto cNaive = zeros(a.length!0, b.length!1);
        auto cMap = zeros(a.length!0, b.length!1);
        auto cDot = zeros(a.length!0, b.length!1);
        auto cParallel = zeros(a.length!0, b.length!1);
        auto cParallelUnroll = zeros(a.length!0, b.length!1);
        auto cGlas = zeros(a.length!0, b.length!1).universal;
        auto cBlas = zeros(a.length!0, b.length!1);

        auto results = benchmark!(
            () { naiveGemm(1, a, b, 0, cNaive); },
            () { mapGemm(1, a, b, 0, cMap); },
            () { dotGemm(1, a, b, 0, cDot); },
            () { dotParallelGemm(1, a, b, 0, cParallel); },
            () { dotParallelUnrollGemm!16(1, a, b, 0, cParallelUnroll); },
            () { G.gemm(1.0, au, bu, 0.0, cGlas); },
            () { gemm(1, a, b, 0, cBlas); }
            )(10);
        writefln!"==== benchmark (m=%d) ===="(m);
        writeln("naive:", results[0]);
        writeln("mapSum:", results[1]);
        writeln("std.numeric.dotProduct:", results[2]);
        writeln("std.parallelism.parallel:", results[3]);
        writeln("unroll:", results[4]);
        writeln("glas:", results[5]);
        writeln("blas:", results[6]);
        enforce(approxEqual(cBlas, cNaive), "blas is invalid");
        enforce(approxEqual(cMap, cNaive), "map is invalid");
        enforce(approxEqual(cDot, cNaive), "dot is invalid");
        enforce(approxEqual(cParallel, cNaive), "parallel is invalid");
        enforce(approxEqual(cGlas, cNaive), "glas is invalid");
        enforce(approxEqual(cParallelUnroll, cNaive), "unroll is invalid");
    }
}
