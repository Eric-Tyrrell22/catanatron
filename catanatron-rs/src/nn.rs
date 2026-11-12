//! Minimal inference for the policy/value MLP trained in `python/train.py`,
//! so search can call the network without leaving Rust.
//!
//! File format (little-endian), written by `python/export.py`:
//!   b"CTNN", u32 version (1 to 4), u32 obs_size, u32 action_size,
//!   u32 num_tensors, then per tensor: u32 rows, u32 cols, rows*cols f32.
//! Tensors, in order: trunk layers (weight, bias)..., policy (weight, bias),
//! value (weight, bias). Weights are PyTorch's [out, in]; biases [1, out].
//! Hidden layers use ReLU.
//!
//! Version 2 adds the spot branch (python/model.py `SpotBranch`) after the
//! tensors: its five index tables (each u32 ndim, ndim u32 dims, then the
//! u32 entries), u32 num_tensors and its tensors as above (tile, node1,
//! node_tiles_in, node2, join, tower, tower_out; biases of zeros where the
//! layer has none), then f32 flat_value_scale. See `SpotBranch`.
//!
//! Weights are held as f16 to halve the cache traffic of a forward pass.
//! Inputs and ReLU activations are mostly zero, so each layer only visits the
//! weight rows of nonzero inputs, accumulating a block of outputs in registers.
//!
//! The arithmetic runs on 8-lane f32 vectors (`simd::F8`): AVX on x86_64,
//! two NEON registers on aarch64. Both use fused multiply-add and
//! round-to-nearest-even f16 conversion in the same order, so results are
//! bit-identical across the two.

use std::io::Read;
use std::sync::Arc;

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "f16c", target_feature = "fma"),
    all(target_arch = "aarch64", target_feature = "neon")
)))]
compile_error!("nn.rs needs x86_64 with F16C and FMA (build with -C target-cpu=native, see .cargo/config.toml) or aarch64 with NEON");

/// Eight f32 lanes and the operations the forward pass needs.
#[cfg(target_arch = "x86_64")]
mod simd {
    use std::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub struct F8(__m256);

    #[inline(always)]
    pub unsafe fn zero() -> F8 {
        F8(_mm256_setzero_ps())
    }
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> F8 {
        F8(_mm256_loadu_ps(p))
    }
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: F8) {
        _mm256_storeu_ps(p, v.0)
    }
    #[inline(always)]
    pub unsafe fn splat(x: f32) -> F8 {
        F8(_mm256_set1_ps(x))
    }
    /// acc + x * w, one rounding.
    #[inline(always)]
    pub unsafe fn fma(acc: F8, x: F8, w: F8) -> F8 {
        F8(_mm256_fmadd_ps(x.0, w.0, acc.0))
    }
    /// Eight f16 values (bits) as f32.
    #[inline(always)]
    pub unsafe fn load_f16(p: *const u16) -> F8 {
        F8(_mm256_cvtph_ps(_mm_loadu_si128(p as *const __m128i)))
    }
    /// Stores `v` as eight f16 values (round to nearest even).
    #[inline(always)]
    pub unsafe fn store_f16(p: *mut u16, v: F8) {
        _mm_storeu_si128(p as *mut __m128i, _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v.0))
    }
    /// Bit `k` set iff lane `k` is > 0.
    #[inline(always)]
    pub unsafe fn positive_mask(v: F8) -> u64 {
        _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(v.0, _mm256_setzero_ps())) as u64
    }
    /// Asks for the cache line at `p` ahead of its use.
    #[inline(always)]
    pub unsafe fn prefetch(p: *const u16) {
        _mm_prefetch::<_MM_HINT_T0>(p as *const i8)
    }
}

/// Eight f32 lanes and the operations the forward pass needs.
#[cfg(target_arch = "aarch64")]
mod simd {
    use std::arch::aarch64::*;
    use std::arch::asm;

    #[derive(Clone, Copy)]
    pub struct F8(float32x4_t, float32x4_t);

    /// Four f16 values (bits) as f32. Inline assembly: the f16 NEON
    /// intrinsics aren't stable yet; FCVTL widens exactly.
    #[inline(always)]
    unsafe fn widen(h: uint16x4_t) -> float32x4_t {
        let out: float32x4_t;
        asm!("fcvtl {o:v}.4s, {i:v}.4h", i = in(vreg) h, o = lateout(vreg) out, options(pure, nomem, nostack));
        out
    }
    /// Four f32 as f16 bits, rounding to nearest even (the default FPCR mode).
    #[inline(always)]
    unsafe fn narrow(f: float32x4_t) -> uint16x4_t {
        let out: uint16x4_t;
        asm!("fcvtn {o:v}.4h, {i:v}.4s", i = in(vreg) f, o = lateout(vreg) out, options(pure, nomem, nostack));
        out
    }

    #[inline(always)]
    pub unsafe fn zero() -> F8 {
        F8(vdupq_n_f32(0.0), vdupq_n_f32(0.0))
    }
    #[inline(always)]
    pub unsafe fn load(p: *const f32) -> F8 {
        F8(vld1q_f32(p), vld1q_f32(p.add(4)))
    }
    #[inline(always)]
    pub unsafe fn store(p: *mut f32, v: F8) {
        vst1q_f32(p, v.0);
        vst1q_f32(p.add(4), v.1);
    }
    #[inline(always)]
    pub unsafe fn splat(x: f32) -> F8 {
        F8(vdupq_n_f32(x), vdupq_n_f32(x))
    }
    /// acc + x * w, one rounding.
    #[inline(always)]
    pub unsafe fn fma(acc: F8, x: F8, w: F8) -> F8 {
        F8(vfmaq_f32(acc.0, x.0, w.0), vfmaq_f32(acc.1, x.1, w.1))
    }
    /// Eight f16 values (bits) as f32.
    #[inline(always)]
    pub unsafe fn load_f16(p: *const u16) -> F8 {
        let h = vld1q_u16(p);
        F8(widen(vget_low_u16(h)), widen(vget_high_u16(h)))
    }
    /// Stores `v` as eight f16 values (round to nearest even).
    #[inline(always)]
    pub unsafe fn store_f16(p: *mut u16, v: F8) {
        vst1q_u16(p, vcombine_u16(narrow(v.0), narrow(v.1)));
    }
    /// Bit `k` set iff lane `k` is > 0.
    #[inline(always)]
    pub unsafe fn positive_mask(v: F8) -> u64 {
        let weights: [u32; 4] = [1, 2, 4, 8];
        let w = vld1q_u32(weights.as_ptr());
        let zero = vdupq_n_f32(0.0);
        let lo = vaddvq_u32(vandq_u32(vcgtq_f32(v.0, zero), w));
        let hi = vaddvq_u32(vandq_u32(vcgtq_f32(v.1, zero), w));
        (lo | hi << 4) as u64
    }
    /// Asks for the cache line at `p` ahead of its use.
    #[inline(always)]
    pub unsafe fn prefetch(p: *const u16) {
        asm!("prfm pldl1keep, [{p}]", p = in(reg) p, options(nostack, readonly, preserves_flags));
    }
}

/// f32 lanes per vector.
const LANES: usize = 8;
/// AVX registers of outputs accumulated at once (TILE * LANES outputs).
const TILE: usize = 8;
/// Output columns per weight block (one register tile).
const BLOCK: usize = TILE * LANES;
/// How a pass over a block prefetches weight rows (they are scattered, so
/// the hardware prefetchers can't predict them): 0 = not at all, 1 = each
/// row's part of the next block as the row is used, 2 = the row PREFETCH
/// entries ahead in the same block.
const PF_MODE: usize = 2;
const PREFETCH: usize = 12;
/// Most positions `PolicyValueNet::forward_batch` evaluates together.
pub const MAX_BATCH: usize = 8;

/// A cache line of f16 weights, so blocks and rows start on line boundaries.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
struct Line([u16; 32]);

#[derive(Clone, Debug)]
struct Layer {
    inputs: usize,
    outputs: usize,
    /// `outputs` rounded up to a multiple of LANES.
    stride: usize,
    /// Columns per weight block: BLOCK, or `stride` when that is smaller.
    block_w: usize,
    /// f16 bits (transposed from PyTorch), in blocks of `block_w` output
    /// columns, the last block possibly narrower: block `t` holds, input-
    /// major, every input's weights to outputs `t * block_w..`, so a zero
    /// input is skipped and a pass over the active inputs of one block
    /// sweeps forward through one contiguous region, the rows a few cache
    /// lines apart. Padding columns are 0.
    weight_t: Vec<Line>,
    /// Padded to `stride` with zeros.
    bias: Vec<f32>,
}

impl Layer {
    /// `w` is PyTorch's [outputs][inputs] f32 weight.
    fn new(inputs: usize, outputs: usize, w: &[f32], bias: Vec<f32>) -> Layer {
        let stride = outputs.div_ceil(LANES) * LANES;
        let block_w = BLOCK.min(stride);
        let total = inputs * stride;
        let mut wt = vec![0f32; total];
        for o in 0..outputs {
            let (t, c) = (o / block_w, o % block_w);
            let w_t = (stride - t * block_w).min(block_w);
            let base = t * inputs * block_w;
            for i in 0..inputs {
                wt[base + i * w_t + c] = w[o * inputs + i];
            }
        }
        let mut weight_t = vec![Line([0; 32]); total.div_ceil(32)];
        // SAFETY: `weight_t` holds at least `total` u16 values, contiguous.
        let dst = unsafe { std::slice::from_raw_parts_mut(weight_t.as_mut_ptr() as *mut u16, total) };
        for (src, dst) in wt.chunks_exact(LANES).zip(dst.chunks_exact_mut(LANES)) {
            // SAFETY: both chunks hold exactly LANES elements.
            unsafe { simd::store_f16(dst.as_mut_ptr(), simd::load(src.as_ptr())) }
        }
        let mut bias = bias;
        bias.resize(stride, 0.0);
        Layer { inputs, outputs, stride, block_w, weight_t, bias }
    }

    /// The weights as f16 bits, output-major (`[outputs][inputs]`, PyTorch's
    /// order), with two zero entries of padding at the end.
    fn rows(&self) -> Vec<u16> {
        let mut rows = vec![0u16; self.outputs * self.inputs + 2];
        for t in 0..self.blocks() {
            let (base, w_t) = self.block(t);
            for i in 0..self.inputs {
                for c in 0..w_t {
                    let o = t * self.block_w + c;
                    if o < self.outputs {
                        // SAFETY: (i, c) lies within block `t`.
                        rows[o * self.inputs + i] = unsafe { *base.add(i * w_t + c) };
                    }
                }
            }
        }
        rows
    }

    /// Number of weight blocks.
    #[inline(always)]
    fn blocks(&self) -> usize {
        self.stride.div_ceil(self.block_w)
    }

    /// Start of block `t` and its width in columns.
    #[inline(always)]
    fn block(&self, t: usize) -> (*const u16, usize) {
        assert!(t < self.blocks());
        let w_t = (self.stride - t * self.block_w).min(self.block_w);
        // SAFETY: block `t` starts within `weight_t` (asserted above).
        let p = unsafe { (self.weight_t.as_ptr() as *const u16).add(t * self.inputs * self.block_w) };
        (p, w_t)
    }

    /// out = bias + W x, where `nz` lists the nonzero entries of x as
    /// (index, value) in increasing index order.
    fn forward(&self, nz: &[(u32, f32)], out: &mut Vec<f32>) {
        assert!(nz.iter().all(|&(i, _)| (i as usize) < self.inputs), "input index out of range");
        out.clear();
        out.resize(self.stride, 0.0);
        for t in 0..self.blocks() {
            let w_t = (self.stride - t * self.block_w).min(self.block_w);
            match w_t / LANES {
                8 => self.tile::<8>(nz, t, out),
                7 => self.tile::<7>(nz, t, out),
                6 => self.tile::<6>(nz, t, out),
                5 => self.tile::<5>(nz, t, out),
                4 => self.tile::<4>(nz, t, out),
                3 => self.tile::<3>(nz, t, out),
                2 => self.tile::<2>(nz, t, out),
                _ => self.tile::<1>(nz, t, out),
            }
        }
        out.truncate(self.outputs);
    }

    /// Block `t`, which is `V` vectors wide, accumulated in registers over
    /// all of `nz`.
    #[inline(always)]
    fn tile<const V: usize>(&self, nz: &[(u32, f32)], t: usize, out: &mut [f32]) {
        let j = t * self.block_w;
        assert!(j + V * LANES <= self.stride && out.len() == self.stride);
        let (base, w_t) = self.block(t);
        assert!(w_t == V * LANES);
        let next = (PF_MODE == 1 && t + 1 < self.blocks()).then(|| self.block(t + 1));
        // SAFETY: every row index is < inputs (asserted in `forward`) and
        // the block is V * LANES wide (asserted above), so all loads/stores
        // stay inside their buffers; prefetches read nothing.
        unsafe {
            let mut acc = [simd::zero(); V];
            for (k, a) in acc.iter_mut().enumerate() {
                *a = simd::load(self.bias.as_ptr().add(j + k * LANES));
            }
            let mut step = |(i, xi): (u32, f32), ahead: Option<*const u16>| {
                if let Some(p) = ahead {
                    simd::prefetch(p);
                    if V * LANES * 2 > 64 {
                        simd::prefetch(p.add(32));
                    }
                }
                let xv = simd::splat(xi);
                let row = base.add(i as usize * (V * LANES));
                for (k, a) in acc.iter_mut().enumerate() {
                    *a = simd::fma(*a, xv, simd::load_f16(row.add(k * LANES)));
                }
            };
            match (PF_MODE, next) {
                (1, Some((nb, nw))) => {
                    for &(i, xi) in nz {
                        let p = nb.add(i as usize * nw);
                        simd::prefetch(p);
                        if nw * 2 > 64 {
                            simd::prefetch(p.add(32));
                        }
                        step((i, xi), None);
                    }
                }
                (2, _) => {
                    let n = nz.len();
                    let main = n.saturating_sub(PREFETCH);
                    for k in 0..main {
                        let ahead = base.add(nz.get_unchecked(k + PREFETCH).0 as usize * (V * LANES));
                        step(*nz.get_unchecked(k), Some(ahead));
                    }
                    for &e in &nz[main..] {
                        step(e, None);
                    }
                }
                _ => {
                    for &e in nz {
                        step(e, None);
                    }
                }
            }
            for (k, a) in acc.iter().enumerate() {
                simd::store(out.as_mut_ptr().add(j + k * LANES), *a);
            }
        }
    }
}

impl Layer {
    /// outs[b] = bias + W x_b for each of the `n` inputs x_b, given as
    /// `xg[u * n + b]` = entry `union[u]` of x_b, where `union` lists (in
    /// increasing order) every input nonzero in any x_b. Each weight row is
    /// loaded once per block of outputs and shared by the whole batch.
    fn forward_batch(&self, union: &[u32], xg: &[f32], n: usize, outs: &mut [Vec<f32>]) {
        assert!((1..=MAX_BATCH).contains(&n) && outs.len() >= n && xg.len() >= union.len() * n);
        assert!(union.iter().all(|&i| (i as usize) < self.inputs), "input index out of range");
        for out in outs[..n].iter_mut() {
            out.clear();
            out.resize(self.stride, 0.0);
        }
        let mut j = 0;
        while j < self.stride {
            // Blocks are multiples of 2 * LANES wide except the last, which
            // ends at `stride`: a pair of vectors never straddles two.
            let vecs = ((self.stride - j) / LANES).min(2);
            let mut b = 0;
            while b < n {
                let m = (n - b).min(4);
                match (vecs, m) {
                    (2, 4) => self.batch_tile::<2, 4>(union, xg, n, b, j, outs),
                    (2, 3) => self.batch_tile::<2, 3>(union, xg, n, b, j, outs),
                    (2, 2) => self.batch_tile::<2, 2>(union, xg, n, b, j, outs),
                    (2, _) => self.batch_tile::<2, 1>(union, xg, n, b, j, outs),
                    (_, 4) => self.batch_tile::<1, 4>(union, xg, n, b, j, outs),
                    (_, 3) => self.batch_tile::<1, 3>(union, xg, n, b, j, outs),
                    (_, 2) => self.batch_tile::<1, 2>(union, xg, n, b, j, outs),
                    _ => self.batch_tile::<1, 1>(union, xg, n, b, j, outs),
                }
                b += m;
            }
            j += vecs * LANES;
        }
        for out in outs[..n].iter_mut() {
            out.truncate(self.outputs);
        }
    }

    /// Outputs `j..j + V * LANES` of inputs `b0..b0 + S`, in registers.
    #[inline(always)]
    fn batch_tile<const V: usize, const S: usize>(
        &self,
        union: &[u32],
        xg: &[f32],
        n: usize,
        b0: usize,
        j: usize,
        outs: &mut [Vec<f32>],
    ) {
        assert!(j + V * LANES <= self.stride && b0 + S <= n);
        assert!(outs[b0..b0 + S].iter().all(|o| o.len() == self.stride));
        let (t, c) = (j / self.block_w, j % self.block_w);
        let (block, w_t) = self.block(t);
        assert!(c + V * LANES <= w_t);
        // SAFETY: row indices are < inputs (asserted in `forward_batch`),
        // columns within the block, xg holds union.len() * n values and
        // every output holds `stride` (asserted above).
        unsafe {
            let base = block.add(c);
            let mut acc = [[simd::zero(); V]; S];
            for row in acc.iter_mut() {
                for (k, a) in row.iter_mut().enumerate() {
                    *a = simd::load(self.bias.as_ptr().add(j + k * LANES));
                }
            }
            let last = union.len().saturating_sub(1);
            for (u, &i) in union.iter().enumerate() {
                simd::prefetch(base.add(*union.get_unchecked((u + PREFETCH).min(last)) as usize * w_t));
                let row = base.add(i as usize * w_t);
                let mut w = [simd::zero(); V];
                for (k, wk) in w.iter_mut().enumerate() {
                    *wk = simd::load_f16(row.add(k * LANES));
                }
                let x = xg.as_ptr().add(u * n + b0);
                for (s, accs) in acc.iter_mut().enumerate() {
                    let xv = simd::splat(*x.add(s));
                    for (a, wk) in accs.iter_mut().zip(&w) {
                        *a = simd::fma(*a, xv, *wk);
                    }
                }
            }
            for (s, accs) in acc.iter().enumerate() {
                let out = outs[b0 + s].as_mut_ptr().add(j);
                for (k, a) in accs.iter().enumerate() {
                    simd::store(out.add(k * LANES), *a);
                }
            }
        }
    }
}

/// Collects the entries of `x` that pass `keep` into `nz` (kept at least
/// `x.len()` long) and returns how many. Branch-free: which inputs are
/// nonzero is unpredictable, so a branch per input would mispredict often.
fn compact(x: &[f32], nz: &mut Vec<(u32, f32)>, keep: impl Fn(f32) -> bool) -> usize {
    if nz.len() < x.len() {
        nz.resize(x.len(), (0, 0.0));
    }
    let mut n = 0;
    for (i, &xi) in x.iter().enumerate() {
        nz[n] = (i as u32, xi);
        n += keep(xi) as usize;
    }
    n
}

/// `compact` with `keep` = x > 0 (the ReLU's survivors): eight lanes are
/// compared at once and only the set bits of the mask are visited, so the
/// cost follows the active entries (a third or less) rather than the width.
fn compact_positive(x: &[f32], nz: &mut Vec<(u32, f32)>) -> usize {
    if nz.len() < x.len() {
        nz.resize(x.len(), (0, 0.0));
    }
    let mut n = 0;
    let mut i = 0;
    // SAFETY: `i + LANES <= x.len()` for every vector load.
    unsafe {
        while i + LANES <= x.len() {
            let mut m = simd::positive_mask(simd::load(x.as_ptr().add(i)));
            while m != 0 {
                let k = i + m.trailing_zeros() as usize;
                *nz.get_unchecked_mut(n) = (k as u32, *x.get_unchecked(k));
                n += 1;
                m &= m - 1;
            }
            i += LANES;
        }
    }
    for k in i..x.len() {
        nz[n] = (k as u32, x[k]);
        n += (x[k] > 0.0) as usize;
    }
    n
}

#[derive(Clone, Debug)]
pub struct PolicyValueNet {
    pub obs_size: usize,
    pub action_size: usize,
    trunk: Vec<Layer>,
    policy: Layer,
    /// The policy weights output-major (`Layer::rows`), for
    /// `forward_sparse_for`.
    policy_rows: Vec<u16>,
    value: Layer,
    spots: Option<SpotBranch>,
    /// The shared per-spot settlement head (version 3), see `SpotHead`.
    spot_head: Option<SpotHead>,
    /// The mid-game board heads (version 4), see `BoardHeads`.
    board_heads: Option<BoardHeads>,
    /// A separate network for the initial placements (`Evaluator` uses it
    /// while `is_initial_build_phase`); it may have its own input layout.
    pub opening: Option<Arc<PolicyValueNet>>,
}

/// Version 3 (python/model.py `SpotHead`): settlement logits from one small
/// network run on every land node. Per node: ReLU(enc(its observation
/// entries) + ctx(trunk's last layer)); then ReLU(mix([a, mean of a over the
/// neighbours at road distance 1, at distance 2])) + a; then `out`, centred
/// over the nodes (the common offset is invisible to the softmax at placements
/// and otherwise drifts), added to the node's settlement logit. Tables: node_feat [N][F] (obs_size = zero
/// slot), adj1 [N][3] and adj2 [N][K] (N = no neighbour), slot [N].
#[derive(Clone, Debug)]
struct SpotHead {
    node_feat: Vec<Vec<u32>>,
    adj1: Vec<Vec<u32>>,
    adj2: Vec<Vec<u32>>,
    slot: Vec<u32>,
    /// Per action slot: the node whose score it carries, or u32::MAX.
    node_of_slot: Vec<u32>,
    enc: Layer,
    ctx: Layer,
    mix: Layer,
    out: Layer,
}

/// Reusable buffers for the spot head.
#[derive(Default)]
struct HeadScratch {
    x: Vec<f32>,
    ctx: Vec<f32>,
    /// [(N + 1) * width]: node activations, the last row zeros (padding).
    a: Vec<f32>,
    input: Vec<f32>,
    tmp: Vec<f32>,
    nz: Vec<(u32, f32)>,
    hidden: Vec<(u32, f32)>,
    scores: Vec<f32>,
    o: Vec<f32>,
    z: Vec<f32>,
}

impl SpotHead {
    /// Node scores into `s.scores` from the observation's nonzero entries and
    /// the trunk's last layer (its ReLU survivors, sparse).
    fn run(&self, obs_nz: &[(u32, f32)], obs_size: usize, hidden: &[(u32, f32)], s: &mut HeadScratch) {
        let HeadScratch { x, ctx, a, input, tmp, nz, scores, o, .. } = s;
        spot_input(obs_nz, obs_size, x);
        self.ctx.forward(hidden, ctx);
        let (n, w) = (self.node_feat.len(), self.enc.outputs);
        a.clear();
        a.resize((n + 1) * w, 0.0);
        for i in 0..n {
            input.clear();
            input.extend(self.node_feat[i].iter().map(|&k| x[k as usize]));
            dense(&self.enc, input, nz, tmp);
            for j in 0..w {
                a[i * w + j] = (tmp[j] + ctx[j]).max(0.0);
            }
        }
        scores.clear();
        for i in 0..n {
            input.clear();
            input.extend_from_slice(&a[i * w..(i + 1) * w]);
            for adj in [&self.adj1[i], &self.adj2[i]] {
                let count = adj.iter().filter(|&&m| (m as usize) < n).count().max(1) as f32;
                let start = input.len();
                input.resize(start + w, 0.0);
                for &m in adj {
                    let row = &a[m as usize * w..(m as usize + 1) * w];
                    for j in 0..w {
                        input[start + j] += row[j];
                    }
                }
                input[start..].iter_mut().for_each(|v| *v /= count);
            }
            dense(&self.mix, input, nz, tmp);
            for j in 0..w {
                tmp[j] = tmp[j].max(0.0) + a[i * w + j];
            }
            dense(&self.out, tmp, nz, o);
            scores.push(o[0]);
        }
        let mean = scores.iter().sum::<f32>() / scores.len().max(1) as f32;
        scores.iter_mut().for_each(|v| *v -= mean);
    }
}

impl PolicyValueNet {
    /// This network with `opening` attached for the initial placements.
    pub fn with_opening(mut self, opening: Arc<PolicyValueNet>) -> Self {
        self.opening = Some(opening);
        self
    }
}

/// Board structure (python/model.py `SpotBranch`): a shared tile encoder and
/// a shared two-layer network per land node, pooled (mean, max) over the
/// nodes. The pooled summary is added to the trunk's second layer before its
/// ReLU (`join`) and feeds a value tower with the global features; the value
/// is `flat_value_scale` * the trunk's value + the tower's.
#[derive(Clone, Debug)]
struct SpotBranch {
    /// Per node: its own observation positions (`obs_size` = a zero slot).
    node_feat: Vec<Vec<u32>>,
    /// Per node, per edge slot: the edge's observation positions.
    node_edge_feat: Vec<Vec<Vec<u32>>>,
    tile_feat: Vec<Vec<u32>>,
    /// Per node: its tiles (`tile_feat.len()` = padding).
    node_tiles: Vec<Vec<u32>>,
    global_feat: Vec<u32>,
    tile: Layer,
    node1: Layer,
    node_tiles_in: Layer,
    node2: Layer,
    join: Layer,
    tower: Layer,
    tower_out: Layer,
    flat_value_scale: f32,
}

/// Reusable buffers for the spot branch.
#[derive(Default)]
struct SpotScratch {
    x: Vec<f32>,
    tiles: Vec<Vec<f32>>,
    input: Vec<f32>,
    sum: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    pooled: Vec<f32>,
    nz: Vec<(u32, f32)>,
}

/// The observation densely, with the zero padding slot at `obs_size`.
fn spot_input(obs_nz: &[(u32, f32)], obs_size: usize, x: &mut Vec<f32>) {
    x.clear();
    x.resize(obs_size + 1, 0.0);
    for &(i, v) in obs_nz {
        x[i as usize] = v;
    }
}

/// Runs `layer` on a dense input (its nonzero entries).
fn dense(layer: &Layer, x: &[f32], nz: &mut Vec<(u32, f32)>, out: &mut Vec<f32>) {
    let n = compact(x, nz, |v| v != 0.0);
    layer.forward(&nz[..n], out);
}

impl SpotBranch {
    /// (join output [hidden], tower value) for an observation given densely
    /// (`x`, with one extra zero at index obs_size).
    fn run(&self, x: &[f32], s: &mut SpotScratch, join_out: &mut Vec<f32>) -> f32 {
        let SpotScratch { tiles, input, sum, a, b, pooled, nz, .. } = s;
        let tw = self.tile.outputs;
        tiles.resize_with(self.tile_feat.len() + 1, Vec::new);
        for (t, feat) in self.tile_feat.iter().enumerate() {
            input.clear();
            input.extend(feat.iter().map(|&i| x[i as usize]));
            dense(&self.tile, input, nz, &mut tiles[t]);
            tiles[t].iter_mut().for_each(|v| *v = v.max(0.0));
        }
        let pad = self.tile_feat.len();
        tiles[pad].clear();
        tiles[pad].resize(tw, 0.0);
        let nw = self.node2.outputs;
        pooled.clear();
        pooled.resize(2 * nw, 0.0);
        let (mean, max) = pooled.split_at_mut(nw);
        max.iter_mut().for_each(|v| *v = f32::NEG_INFINITY);
        for n in 0..self.node_feat.len() {
            input.clear();
            input.extend(self.node_feat[n].iter().map(|&i| x[i as usize]));
            let f_edge = self.node_edge_feat[n][0].len();
            for f in 0..f_edge {
                input.push(self.node_edge_feat[n].iter().map(|e| x[e[f] as usize]).sum());
            }
            dense(&self.node1, input, nz, a);
            sum.clear();
            sum.resize(tw, 0.0);
            for &t in &self.node_tiles[n] {
                for (acc, v) in sum.iter_mut().zip(&tiles[t as usize]) {
                    *acc += v;
                }
            }
            dense(&self.node_tiles_in, sum, nz, b);
            for (h, v) in a.iter_mut().zip(b.iter()) {
                *h = (*h + v).max(0.0);
            }
            dense(&self.node2, a, nz, b);
            for ((m, mx), v) in mean.iter_mut().zip(max.iter_mut()).zip(b.iter()) {
                let v = v.max(0.0);
                *m += v;
                *mx = mx.max(v);
            }
        }
        let count = self.node_feat.len() as f32;
        mean.iter_mut().for_each(|m| *m /= count);
        dense(&self.join, pooled, nz, join_out);
        input.clear();
        input.extend_from_slice(pooled);
        input.extend(self.global_feat.iter().map(|&i| x[i as usize]));
        dense(&self.tower, input, nz, a);
        a.iter_mut().for_each(|v| *v = v.max(0.0));
        dense(&self.tower_out, a, nz, b);
        b[0]
    }
}

/// Version 4 (python/model.py `BoardHeads`): the spot head's node encoder (enc, ctx, mix) gives an
/// embedding z per land node; settle_out / city_out score each node; edge_enc + edge_out score a land
/// edge from [z_a + z_b, |z_a - z_b|, its EDGE entries]; tile_enc + tile_out score a tile from [mean z of
/// its nodes, its TILE entries]. Each kind's scores are centred and added to its slots (a tile's score
/// to every victim slot). Tables as the spot head's plus edge_nodes [E][2], edge_feat [E][Fe],
/// tile_nodes [T][K] (N = none), tile_feat [T][Ft], city_slot [N], road_slot [E], robber_slot [T][V].
/// A small f32 layer applied to many rows at once (the board heads: 54 nodes, 72 edges, 19 tiles per
/// forward). Weights input-major (`w_t[i][o]`), so every inner loop is an independent-lane update over
/// the outputs that the compiler vectorises (a dot-product reduction would not be), and zero inputs
/// (one-hots are mostly zero) cost nothing.
#[derive(Clone, Debug)]
struct Dense {
    inputs: usize,
    outputs: usize,
    w_t: Vec<f32>,
    b: Vec<f32>,
}

impl Dense {
    fn new(inputs: usize, outputs: usize, w: &[f32], b: Vec<f32>) -> Dense {
        let mut w_t = vec![0f32; inputs * outputs];
        for o in 0..outputs {
            for i in 0..inputs {
                w_t[i * outputs + o] = w[o * inputs + i];
            }
        }
        Dense { inputs, outputs, w_t, b }
    }

    /// `out[r][o] = b[o] + sum_i x[r][i] * w[o][i]` for `rows` rows of `x` (row stride `inputs`).
    fn forward_rows(&self, x: &[f32], rows: usize, out: &mut Vec<f32>) {
        debug_assert_eq!(x.len(), rows * self.inputs);
        let (ni, no) = (self.inputs, self.outputs);
        out.clear();
        out.reserve(rows * no);
        for r in 0..rows {
            out.extend_from_slice(&self.b);
            let o = &mut out[r * no..(r + 1) * no];
            for (k, &xk) in x[r * ni..(r + 1) * ni].iter().enumerate() {
                if xk != 0.0 {
                    let wk = &self.w_t[k * no..(k + 1) * no];
                    for j in 0..no {
                        o[j] += xk * wk[j];
                    }
                }
            }
        }
    }

    /// `out[o] = b[o] + sum over the sparse input of x_i * w[o][i]`.
    fn forward_sparse(&self, x: &[(u32, f32)], out: &mut Vec<f32>) {
        out.clear();
        out.extend_from_slice(&self.b);
        let no = self.outputs;
        for &(i, v) in x {
            let wk = &self.w_t[i as usize * no..(i as usize + 1) * no];
            for j in 0..no {
                out[j] += v * wk[j];
            }
        }
    }
}

#[derive(Clone, Debug)]
struct BoardHeads {
    node_feat: Vec<Vec<u32>>,
    adj1: Vec<Vec<u32>>,
    adj2: Vec<Vec<u32>>,
    edge_nodes: Vec<Vec<u32>>,
    edge_feat: Vec<Vec<u32>>,
    tile_nodes: Vec<Vec<u32>>,
    tile_feat: Vec<Vec<u32>>,
    /// Per tile: its victim slots (the tile's score goes to each).
    robber_slot: Vec<Vec<u32>>,
    /// Per action slot: its index in the scores this produces (`run` fills `s.out` in that order), or u32::MAX.
    index_of_slot: Vec<u32>,
    /// The slots in `out` order.
    slots: Vec<u32>,
    enc: Dense,
    ctx: Dense,
    mix: Dense,
    settle_out: Dense,
    city_out: Dense,
    edge_enc: Dense,
    edge_out: Dense,
    tile_enc: Dense,
    tile_out: Dense,
}

impl BoardHeads {
    /// The additions per slot of `self.slots`, into `s.scores`, from the observation's nonzero entries
    /// and the trunk's last layer (sparse). Dense f32 passes over all nodes / edges / tiles at once.
    fn run(&self, obs_nz: &[(u32, f32)], obs_size: usize, hidden: &[(u32, f32)], s: &mut HeadScratch) {
        let HeadScratch { x, ctx, a, input, tmp, scores, o, z, .. } = s;
        spot_input(obs_nz, obs_size, x);
        self.ctx.forward_sparse(hidden, ctx);
        let (n, w) = (self.node_feat.len(), self.enc.outputs);
        // Node encoder over all nodes: gather [n][f], one pass, + ctx, ReLU; row n stays zero (padding).
        input.clear();
        for feat in &self.node_feat {
            input.extend(feat.iter().map(|&k| x[k as usize]));
        }
        self.enc.forward_rows(input, n, tmp);
        a.clear();
        a.resize((n + 1) * w, 0.0);
        for i in 0..n {
            for j in 0..w {
                a[i * w + j] = (tmp[i * w + j] + ctx[j]).max(0.0);
            }
        }
        // Message passing: [a, mean over adj1, mean over adj2] -> mix -> ReLU, + a.
        input.clear();
        input.resize(n * 3 * w, 0.0);
        for i in 0..n {
            let row = &mut input[i * 3 * w..(i + 1) * 3 * w];
            row[..w].copy_from_slice(&a[i * w..(i + 1) * w]);
            for (k, adj) in [&self.adj1[i], &self.adj2[i]].into_iter().enumerate() {
                let count = adj.iter().filter(|&&m| (m as usize) < n).count().max(1) as f32;
                let dst = &mut row[(k + 1) * w..(k + 2) * w];
                for &m in adj {
                    let src = &a[m as usize * w..(m as usize + 1) * w];
                    for j in 0..w {
                        dst[j] += src[j];
                    }
                }
                dst.iter_mut().for_each(|v| *v /= count);
            }
        }
        self.mix.forward_rows(input, n, tmp);
        z.clear();
        z.resize((n + 1) * w, 0.0);
        for i in 0..n * w {
            z[i] = tmp[i].max(0.0) + a[i];
        }
        scores.clear();
        let centred = |scores: &mut Vec<f32>, start: usize| {
            let mean = scores[start..].iter().sum::<f32>() / (scores.len() - start).max(1) as f32;
            scores[start..].iter_mut().for_each(|v| *v -= mean);
        };
        // Settlements, then cities, per node.
        for head in [&self.settle_out, &self.city_out] {
            let start = scores.len();
            head.forward_rows(&z[..n * w], n, o);
            scores.extend_from_slice(o);
            centred(scores, start);
        }
        // Roads: [z_a + z_b, |z_a - z_b|, edge entries] per land edge.
        let (ne, fe) = (self.edge_nodes.len(), self.edge_feat.first().map_or(0, |f| f.len()));
        input.clear();
        for (e, nodes) in self.edge_nodes.iter().enumerate() {
            let (p, q) = (nodes[0] as usize * w, nodes[1] as usize * w);
            for j in 0..w {
                input.push(z[p + j] + z[q + j]);
            }
            for j in 0..w {
                input.push((z[p + j] - z[q + j]).abs());
            }
            input.extend(self.edge_feat[e].iter().map(|&k| x[k as usize]));
        }
        debug_assert_eq!(input.len(), ne * (2 * w + fe));
        self.edge_enc.forward_rows(input, ne, tmp);
        tmp.iter_mut().for_each(|v| *v = v.max(0.0));
        let start = scores.len();
        self.edge_out.forward_rows(tmp, ne, o);
        scores.extend_from_slice(o);
        centred(scores, start);
        // Robber: [mean z over the tile's nodes, tile entries] per tile; one score per victim slot.
        let nt = self.tile_nodes.len();
        input.clear();
        for (t, nodes) in self.tile_nodes.iter().enumerate() {
            let count = nodes.iter().filter(|&&m| (m as usize) < n).count().max(1) as f32;
            let start = input.len();
            input.resize(start + w, 0.0);
            for &m in nodes {
                let src = &z[m as usize * w..(m as usize + 1) * w]; // row n is zeros
                for j in 0..w {
                    input[start + j] += src[j];
                }
            }
            input[start..].iter_mut().for_each(|v| *v /= count);
            input.extend(self.tile_feat[t].iter().map(|&k| x[k as usize]));
        }
        self.tile_enc.forward_rows(input, nt, tmp);
        tmp.iter_mut().for_each(|v| *v = v.max(0.0));
        self.tile_out.forward_rows(tmp, nt, o);
        let mean = o.iter().sum::<f32>() / nt.max(1) as f32;
        for (t, slots) in self.robber_slot.iter().enumerate() {
            for _ in slots {
                scores.push(o[t] - mean);
            }
        }
        debug_assert_eq!(scores.len(), self.slots.len());
    }
}

/// Reusable scratch buffers for `PolicyValueNet::forward`.
#[derive(Default)]
pub struct Scratch {
    a: Vec<f32>,
    v: Vec<f32>,
    nz: Vec<(u32, f32)>,
    obs_nz: Vec<(u32, f32)>,
    spot: SpotScratch,
    join: Vec<f32>,
    head: HeadScratch,
}

/// Reusable buffers for `PolicyValueNet::forward_batch`.
#[derive(Default)]
pub struct BatchScratch {
    /// Per position: layer outputs.
    h: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    union: Vec<u32>,
    xg: Vec<f32>,
    /// Bit per input: in the union. Kept all-zero between uses.
    bits: Vec<u64>,
    /// Per input: position in `union`, valid for union members.
    slot: Vec<u32>,
    spot: SpotScratch,
    /// Per position: the spot branch's join output and tower value.
    join: Vec<Vec<f32>>,
    tower: Vec<f32>,
    head: HeadScratch,
}

impl BatchScratch {
    /// Moves the set bits of `bits` (clearing them) into `union`, in order.
    fn drain_bits(&mut self) {
        self.union.clear();
        for (w, word) in self.bits.iter_mut().enumerate() {
            let mut x = std::mem::take(word);
            while x != 0 {
                self.union.push((w * 64) as u32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }

    /// Union of the positions' sparse inputs and their gathered values.
    fn gather_sparse(&mut self, xs: &[&[(u32, f32)]], inputs: usize) {
        let n = xs.len();
        self.bits.clear();
        self.bits.resize(inputs.div_ceil(64), 0);
        if self.slot.len() < inputs {
            self.slot.resize(inputs, 0);
        }
        for x in xs {
            for &(i, _) in x.iter() {
                self.bits[i as usize / 64] |= 1 << (i % 64);
            }
        }
        self.drain_bits();
        for (u, &i) in self.union.iter().enumerate() {
            self.slot[i as usize] = u as u32;
        }
        self.xg.clear();
        self.xg.resize(self.union.len() * n, 0.0);
        for (b, x) in xs.iter().enumerate() {
            for &(i, v) in x.iter() {
                self.xg[self.slot[i as usize] as usize * n + b] = v;
            }
        }
    }

    /// Union of the positive entries (ReLU) of `h[..n]`, and their values.
    fn gather_relu(&mut self, n: usize) {
        let width = self.h[0].len();
        self.bits.clear();
        self.bits.resize(width.div_ceil(64), 0);
        let mut i = 0;
        // SAFETY: each `h[b]` holds `width` values and `i + LANES <= width`.
        unsafe {
            while i + LANES <= width {
                let mut m = 0u64;
                for h in &self.h[..n] {
                    m |= simd::positive_mask(simd::load(h.as_ptr().add(i)));
                }
                self.bits[i / 64] |= m << (i % 64);
                i += LANES;
            }
        }
        for k in i..width {
            if self.h[..n].iter().any(|h| h[k] > 0.0) {
                self.bits[k / 64] |= 1 << (k % 64);
            }
        }
        self.drain_bits();
        self.xg.clear();
        self.xg.resize(self.union.len() * n, 0.0);
        for (u, &i) in self.union.iter().enumerate() {
            for b in 0..n {
                self.xg[u * n + b] = self.h[b][i as usize].max(0.0);
            }
        }
    }
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_f32(r: &mut impl Read) -> std::io::Result<f32> {
    Ok(f32::from_bits(read_u32(r)?))
}

fn bad(msg: String) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

/// `n` tensors (weight, bias pairs) as layers.
fn read_layers(r: &mut impl Read, n: usize) -> std::io::Result<Vec<Layer>> {
    let mut tensors = Vec::with_capacity(n);
    for _ in 0..n {
        let rows = read_u32(r)? as usize;
        let cols = read_u32(r)? as usize;
        let mut data = vec![0f32; rows * cols];
        for x in data.iter_mut() {
            *x = read_f32(r)?;
        }
        tensors.push((rows, cols, data));
    }
    let mut layers = Vec::new();
    let mut it = tensors.into_iter();
    while let (Some((out, inp, w)), Some((_, bout, b))) = (it.next(), it.next()) {
        if bout != out {
            return Err(bad(format!("bias length {bout} doesn't match {out} outputs")));
        }
        layers.push(Layer::new(inp, out, &w, b));
    }
    Ok(layers)
}

/// `n` tensors (weight, bias pairs) as dense f32 layers.
fn read_dense(r: &mut impl Read, n: usize) -> std::io::Result<Vec<Dense>> {
    let mut tensors = Vec::with_capacity(n);
    for _ in 0..n {
        let rows = read_u32(r)? as usize;
        let cols = read_u32(r)? as usize;
        let mut data = vec![0f32; rows * cols];
        for x in data.iter_mut() {
            *x = read_f32(r)?;
        }
        tensors.push((rows, cols, data));
    }
    let mut layers = Vec::new();
    let mut it = tensors.into_iter();
    while let (Some((out, inp, w)), Some((_, bout, b))) = (it.next(), it.next()) {
        if bout != out {
            return Err(bad(format!("bias length {bout} doesn't match {out} outputs")));
        }
        layers.push(Dense::new(inp, out, &w, b));
    }
    Ok(layers)
}

/// An index table: (dims, entries).
fn read_table(r: &mut impl Read) -> std::io::Result<(Vec<usize>, Vec<u32>)> {
    let ndim = read_u32(r)? as usize;
    let dims = (0..ndim).map(|_| read_u32(r).map(|d| d as usize)).collect::<std::io::Result<Vec<_>>>()?;
    let len = dims.iter().product::<usize>();
    let data = (0..len).map(|_| read_u32(r)).collect::<std::io::Result<Vec<_>>>()?;
    Ok((dims, data))
}

fn rows(dims: &[usize], data: &[u32], want: usize) -> std::io::Result<Vec<Vec<u32>>> {
    if dims.len() != want {
        return Err(bad(format!("index table has {} dims, expected {want}", dims.len())));
    }
    let w = dims[1..].iter().product::<usize>().max(1);
    Ok(data.chunks(w).map(|c| c.to_vec()).collect())
}

fn read_spots(r: &mut impl Read, obs_size: usize) -> std::io::Result<SpotBranch> {
    let (d, node_feat) = read_table(r)?;
    let node_feat = rows(&d, &node_feat, 2)?;
    let (d, ef) = read_table(r)?;
    if d.len() != 3 {
        return Err(bad("node_edge_feat needs 3 dims".into()));
    }
    let node_edge_feat = ef.chunks(d[1] * d[2]).map(|n| n.chunks(d[2]).map(|e| e.to_vec()).collect()).collect();
    let (d, tf) = read_table(r)?;
    let tile_feat = rows(&d, &tf, 2)?;
    let (d, nt) = read_table(r)?;
    let node_tiles = rows(&d, &nt, 2)?;
    let (d, global_feat) = read_table(r)?;
    if d.len() != 1 {
        return Err(bad("global_feat needs 1 dim".into()));
    }
    let n = read_u32(r)? as usize;
    if n != 14 {
        return Err(bad(format!("spot branch has {n} tensors, expected 14")));
    }
    let mut l = read_layers(r, n)?.into_iter();
    let mut next = || l.next().unwrap();
    let (tile, node1, node_tiles_in, node2, join, tower, tower_out) = (next(), next(), next(), next(), next(), next(), next());
    let flat_value_scale = read_f32(r)?;
    let in_range = |v: &u32| (*v as usize) <= obs_size;
    if !node_feat.iter().flatten().all(in_range) || !tile_feat.iter().flatten().all(in_range) || !global_feat.iter().all(in_range) {
        return Err(bad("spot index out of range".into()));
    }
    Ok(SpotBranch { node_feat, node_edge_feat, tile_feat, node_tiles, global_feat, tile, node1, node_tiles_in, node2, join, tower, tower_out, flat_value_scale })
}

fn read_spot_head(r: &mut impl Read, obs_size: usize, action_size: usize) -> std::io::Result<SpotHead> {
    let (d, nf) = read_table(r)?;
    let node_feat = rows(&d, &nf, 2)?;
    let (d, a1) = read_table(r)?;
    let adj1 = rows(&d, &a1, 2)?;
    let (d, a2) = read_table(r)?;
    let adj2 = rows(&d, &a2, 2)?;
    let (d, slot) = read_table(r)?;
    if d.len() != 1 {
        return Err(bad("slot table needs 1 dim".into()));
    }
    let n = read_u32(r)? as usize;
    if n != 8 {
        return Err(bad(format!("spot head has {n} tensors, expected 8")));
    }
    let mut l = read_layers(r, n)?.into_iter();
    let mut next = || l.next().unwrap();
    let (enc, ctx, mix, out) = (next(), next(), next(), next());
    let nodes = node_feat.len();
    if adj1.len() != nodes || adj2.len() != nodes || slot.len() != nodes {
        return Err(bad("spot head tables disagree on the node count".into()));
    }
    if !node_feat.iter().flatten().all(|&v| (v as usize) <= obs_size) || !adj1.iter().chain(&adj2).flatten().all(|&v| (v as usize) <= nodes)
        || !slot.iter().all(|&v| (v as usize) < action_size) {
        return Err(bad("spot head index out of range".into()));
    }
    let w = enc.outputs;
    if ctx.outputs != w || mix.inputs != 3 * w || mix.outputs != w || out.inputs != w || out.outputs != 1 || node_feat.iter().any(|f| f.len() != enc.inputs) {
        return Err(bad("spot head shapes don't match".into()));
    }
    let mut node_of_slot = vec![u32::MAX; action_size];
    for (i, &s) in slot.iter().enumerate() {
        node_of_slot[s as usize] = i as u32;
    }
    Ok(SpotHead { node_feat, adj1, adj2, slot, node_of_slot, enc, ctx, mix, out })
}

fn read_board_heads(r: &mut impl Read, obs_size: usize, action_size: usize) -> std::io::Result<BoardHeads> {
    fn t2(r: &mut impl Read) -> std::io::Result<Vec<Vec<u32>>> {
        let (d, v) = read_table(r)?;
        rows(&d, &v, 2)
    }
    fn t1(r: &mut impl Read) -> std::io::Result<Vec<u32>> {
        let (d, v) = read_table(r)?;
        if d.len() != 1 {
            return Err(bad("expected a 1-dim table".into()));
        }
        Ok(v)
    }
    let node_feat = t2(r)?;
    let adj1 = t2(r)?;
    let adj2 = t2(r)?;
    let settle_slot = t1(r)?;
    let edge_nodes = t2(r)?;
    let edge_feat = t2(r)?;
    let tile_nodes = t2(r)?;
    let tile_feat = t2(r)?;
    let city_slot = t1(r)?;
    let road_slot = t1(r)?;
    let robber_slot = t2(r)?;
    let n = read_u32(r)? as usize;
    if n != 18 {
        return Err(bad(format!("board heads have {n} tensors, expected 18")));
    }
    let mut l = read_dense(r, n)?.into_iter();
    let mut next = || l.next().unwrap();
    let (enc, ctx, mix, settle_out, city_out, edge_enc, edge_out, tile_enc, tile_out) = (next(), next(), next(), next(), next(), next(), next(), next(), next());
    let nodes = node_feat.len();
    let w = enc.outputs;
    let ok_shapes = adj1.len() == nodes && adj2.len() == nodes && settle_slot.len() == nodes && city_slot.len() == nodes
        && edge_nodes.len() == edge_feat.len() && road_slot.len() == edge_nodes.len() && tile_nodes.len() == tile_feat.len() && robber_slot.len() == tile_nodes.len()
        && ctx.outputs == w && mix.inputs == 3 * w && mix.outputs == w && settle_out.inputs == w && city_out.inputs == w
        && edge_enc.inputs == 2 * w + edge_feat.first().map_or(0, |f| f.len()) && edge_enc.outputs == w && edge_out.inputs == w
        && tile_enc.inputs == w + tile_feat.first().map_or(0, |f| f.len()) && tile_enc.outputs == w && tile_out.inputs == w
        && [&settle_out, &city_out, &edge_out, &tile_out].iter().all(|l| l.outputs == 1) && node_feat.iter().all(|f| f.len() == enc.inputs)
        && edge_nodes.iter().all(|e| e.len() == 2);
    if !ok_shapes {
        return Err(bad("board head shapes don't match".into()));
    }
    let in_obs = |v: &u32| (*v as usize) <= obs_size;
    if !node_feat.iter().chain(&edge_feat).chain(&tile_feat).flatten().all(in_obs)
        || !adj1.iter().chain(&adj2).chain(&tile_nodes).flatten().all(|&v| (v as usize) <= nodes)
        || !edge_nodes.iter().flatten().all(|&v| (v as usize) < nodes)
        || !settle_slot.iter().chain(&city_slot).chain(&road_slot).chain(robber_slot.iter().flatten()).all(|&v| (v as usize) < action_size) {
        return Err(bad("board head index out of range".into()));
    }
    let slots: Vec<u32> = settle_slot.iter().chain(&city_slot).chain(&road_slot).chain(robber_slot.iter().flatten()).copied().collect();
    let mut index_of_slot = vec![u32::MAX; action_size];
    for (i, &s) in slots.iter().enumerate() {
        if index_of_slot[s as usize] != u32::MAX {
            return Err(bad("a slot appears twice in the board heads' tables".into()));
        }
        index_of_slot[s as usize] = i as u32;
    }
    Ok(BoardHeads { node_feat, adj1, adj2, edge_nodes, edge_feat, tile_nodes, tile_feat, robber_slot, index_of_slot, slots,
                    enc, ctx, mix, settle_out, city_out, edge_enc, edge_out, tile_enc, tile_out })
}

impl PolicyValueNet {
    pub fn load(path: &str) -> Result<PolicyValueNet, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        Self::from_bytes(&bytes).map_err(|e| format!("{path}: {e}"))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<PolicyValueNet, String> {
        let mut r = bytes;
        let io = |e: std::io::Error| e.to_string();
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic).map_err(io)?;
        if &magic != b"CTNN" {
            return Err("not a CTNN network file".into());
        }
        let version = read_u32(&mut r).map_err(io)?;
        if !(1..=4).contains(&version) {
            return Err(format!("unsupported network file version {version}"));
        }
        let obs_size = read_u32(&mut r).map_err(io)? as usize;
        let action_size = read_u32(&mut r).map_err(io)? as usize;
        let n = read_u32(&mut r).map_err(io)? as usize;
        if n < 6 || n % 2 != 0 {
            return Err(format!("expected an even number (>= 6) of tensors, got {n}"));
        }
        let mut layers = read_layers(&mut r, n).map_err(io)?;
        let spots = if version == 2 { Some(read_spots(&mut r, obs_size).map_err(io)?) } else { None };
        let spot_head = if version == 3 { Some(read_spot_head(&mut r, obs_size, action_size).map_err(io)?) } else { None };
        let board_heads = if version == 4 { Some(read_board_heads(&mut r, obs_size, action_size).map_err(io)?) } else { None };
        if !r.is_empty() {
            return Err("trailing bytes after tensors".into());
        }
        let value = layers.pop().unwrap();
        let policy = layers.pop().unwrap();
        let trunk = layers;
        let mut width = obs_size;
        for l in &trunk {
            if l.inputs != width {
                return Err(format!("layer expects {} inputs, previous gives {width}", l.inputs));
            }
            width = l.outputs;
        }
        if policy.inputs != width || value.inputs != width || policy.outputs != action_size || value.outputs != 1 {
            return Err("head shapes don't match the trunk/action size".into());
        }
        if let Some(sp) = &spots {
            if trunk.len() < 2 || sp.join.outputs != trunk[1].outputs || sp.tower_out.outputs != 1 {
                return Err("spot branch shapes don't match the trunk".into());
            }
        }
        if let Some(h) = &spot_head {
            if h.ctx.inputs != width {
                return Err("spot head context doesn't match the trunk width".into());
            }
        }
        if let Some(h) = &board_heads {
            if h.ctx.inputs != width {
                return Err("board heads' context doesn't match the trunk width".into());
            }
        }
        let policy_rows = policy.rows();
        Ok(PolicyValueNet { obs_size, action_size, trunk, policy, policy_rows, value, spots, spot_head, board_heads, opening: None })
    }

    /// Policy logits (unmasked) into `logits`; returns the value estimate.
    pub fn forward(&self, obs: &[f32], scratch: &mut Scratch, logits: &mut Vec<f32>) -> f32 {
        assert_eq!(obs.len(), self.obs_size, "observation size mismatch");
        let mut obs_nz = std::mem::take(&mut scratch.obs_nz);
        let n = compact(obs, &mut obs_nz, |x| x != 0.0);
        let value = self.forward_sparse(&obs_nz[..n], scratch, logits);
        scratch.obs_nz = obs_nz;
        value
    }

    /// `forward_sparse` for up to MAX_BATCH positions at once, sharing each
    /// weight row between them: logits into `logits[b]`, value into
    /// `values[b]`. Same results as one `forward_sparse` per position.
    pub fn forward_batch(
        &self,
        obs_nz: &[&[(u32, f32)]],
        scratch: &mut BatchScratch,
        logits: &mut [Vec<f32>],
        values: &mut [f32],
    ) {
        let n = obs_nz.len();
        assert!((1..=MAX_BATCH).contains(&n) && logits.len() >= n && values.len() >= n);
        if scratch.h.len() < MAX_BATCH {
            scratch.h.resize_with(MAX_BATCH, Vec::new);
            scratch.v.resize_with(MAX_BATCH, Vec::new);
        }
        if let Some(sp) = &self.spots {
            scratch.join.resize_with(n.max(scratch.join.len()), Vec::new);
            scratch.tower.resize(n, 0.0);
            for (b, x) in obs_nz.iter().enumerate() {
                let mut dense_x = std::mem::take(&mut scratch.spot.x);
                spot_input(x, self.obs_size, &mut dense_x);
                scratch.tower[b] = sp.run(&dense_x, &mut scratch.spot, &mut scratch.join[b]);
                scratch.spot.x = dense_x;
            }
        }
        scratch.gather_sparse(obs_nz, self.obs_size);
        for (k, layer) in self.trunk.iter().enumerate() {
            layer.forward_batch(&scratch.union, &scratch.xg, n, &mut scratch.h);
            if k == 1 && self.spots.is_some() {
                for (h, j) in scratch.h[..n].iter_mut().zip(&scratch.join[..n]) {
                    h.iter_mut().zip(j).for_each(|(x, y)| *x += y);
                }
            }
            scratch.gather_relu(n);
        }
        self.policy.forward_batch(&scratch.union, &scratch.xg, n, logits);
        if let Some(h) = &self.spot_head {
            // Each position's hidden layer (its positive entries) from the gathered union.
            for b in 0..n {
                scratch.head.hidden.clear();
                for (u, &i) in scratch.union.iter().enumerate() {
                    let v = scratch.xg[u * n + b];
                    if v > 0.0 {
                        scratch.head.hidden.push((i, v));
                    }
                }
                let hidden = std::mem::take(&mut scratch.head.hidden);
                h.run(obs_nz[b], self.obs_size, &hidden, &mut scratch.head);
                scratch.head.hidden = hidden;
                for (&s, &v) in h.slot.iter().zip(&scratch.head.scores) {
                    logits[b][s as usize] += v;
                }
            }
        }
        if let Some(h) = &self.board_heads {
            for b in 0..n {
                scratch.head.hidden.clear();
                for (u, &i) in scratch.union.iter().enumerate() {
                    let v = scratch.xg[u * n + b];
                    if v > 0.0 {
                        scratch.head.hidden.push((i, v));
                    }
                }
                let hidden = std::mem::take(&mut scratch.head.hidden);
                h.run(obs_nz[b], self.obs_size, &hidden, &mut scratch.head);
                scratch.head.hidden = hidden;
                for (&s, &v) in h.slots.iter().zip(&scratch.head.scores) {
                    logits[b][s as usize] += v;
                }
            }
        }
        self.value.forward_batch(&scratch.union, &scratch.xg, n, &mut scratch.v);
        for (b, (out, v)) in values.iter_mut().zip(&scratch.v[..n]).enumerate() {
            *out = match &self.spots {
                Some(sp) => sp.flat_value_scale * v[0] + scratch.tower[b],
                None => v[0],
            };
        }
    }

    /// Like `forward`, from the observation's nonzero entries as (index,
    /// value) in increasing index order (see `ObservationSpace::encode_sparse_into`).
    pub fn forward_sparse(&self, obs_nz: &[(u32, f32)], scratch: &mut Scratch, logits: &mut Vec<f32>) -> f32 {
        let (n, tower) = self.trunk_sparse(obs_nz, scratch);
        self.policy.forward(&scratch.nz[..n], logits);
        if let Some(h) = &self.spot_head {
            let Scratch { nz, head, .. } = &mut *scratch;
            h.run(obs_nz, self.obs_size, &nz[..n], head);
            for (&s, &v) in h.slot.iter().zip(&head.scores) {
                logits[s as usize] += v;
            }
        }
        if let Some(h) = &self.board_heads {
            let Scratch { nz, head, .. } = &mut *scratch;
            h.run(obs_nz, self.obs_size, &nz[..n], head);
            for (&s, &v) in h.slots.iter().zip(&head.scores) {
                logits[s as usize] += v;
            }
        }
        self.value(n, tower, scratch)
    }

    /// `forward_sparse`, but only the logits of action slots `slots` (in
    /// that order, `logits[k]` for `slots[k]`; a slot of `u32::MAX` gets
    /// -inf). Each logit is accumulated in the same order as by the full
    /// policy layer, so the values are identical; the work follows the
    /// number of slots (the legal moves) rather than the action space.
    pub fn forward_sparse_for(&self, obs_nz: &[(u32, f32)], slots: &[u32], scratch: &mut Scratch, logits: &mut Vec<f32>) -> f32 {
        let (n, tower) = self.trunk_sparse(obs_nz, scratch);
        self.policy_subset(&scratch.nz[..n], slots, logits);
        if let Some(h) = &self.spot_head {
            if slots.iter().any(|&s| s != u32::MAX && h.node_of_slot[s as usize] != u32::MAX) {
                let Scratch { nz, head, .. } = &mut *scratch;
                h.run(obs_nz, self.obs_size, &nz[..n], head);
                for (l, &s) in logits.iter_mut().zip(slots) {
                    if s != u32::MAX {
                        let node = h.node_of_slot[s as usize];
                        if node != u32::MAX {
                            *l += head.scores[node as usize];
                        }
                    }
                }
            }
        }
        if let Some(h) = &self.board_heads {
            if slots.iter().any(|&s| s != u32::MAX && h.index_of_slot[s as usize] != u32::MAX) {
                let Scratch { nz, head, .. } = &mut *scratch;
                h.run(obs_nz, self.obs_size, &nz[..n], head);
                for (l, &s) in logits.iter_mut().zip(slots) {
                    if s != u32::MAX {
                        let i = h.index_of_slot[s as usize];
                        if i != u32::MAX {
                            *l += head.scores[i as usize];
                        }
                    }
                }
            }
        }
        self.value(n, tower, scratch)
    }

    /// Runs the trunk (and spot branch): the ReLU survivors of the last
    /// layer are `scratch.nz[..n]`; returns (n, the tower value if any).
    fn trunk_sparse(&self, obs_nz: &[(u32, f32)], scratch: &mut Scratch) -> (usize, Option<f32>) {
        let Scratch { a, nz, spot, join, .. } = scratch;
        let tower = self.spots.as_ref().map(|sp| {
            let mut x = std::mem::take(&mut spot.x);
            spot_input(obs_nz, self.obs_size, &mut x);
            let t = sp.run(&x, spot, join);
            spot.x = x;
            t
        });
        let (first, rest) = self.trunk.split_first().expect("network has a trunk layer");
        first.forward(obs_nz, a);
        // ReLU: only positive activations reach the next layer.
        let mut n = compact_positive(a, nz);
        for (k, layer) in rest.iter().enumerate() {
            layer.forward(&nz[..n], a);
            if k == 0 && tower.is_some() {
                // Trunk layer 2: the spot branch joins before the ReLU.
                a.iter_mut().zip(join.iter()).for_each(|(x, y)| *x += y);
            }
            n = compact_positive(a, nz);
        }
        (n, tower)
    }

    /// The value head on `scratch.nz[..n]`, combined with the tower's value.
    fn value(&self, n: usize, tower: Option<f32>, scratch: &mut Scratch) -> f32 {
        let Scratch { v, nz, .. } = scratch;
        self.value.forward(&nz[..n], v);
        match (&self.spots, tower) {
            (Some(sp), Some(t)) => sp.flat_value_scale * v[0] + t,
            _ => v[0],
        }
    }

    /// Policy logits for `slots` only, into `out` (see `forward_sparse_for`):
    /// eight slots at a time, each lane accumulating bias + sum over `nz` of
    /// x_i * W[slot][i] with one fused multiply-add per term, in `nz` order:
    /// the same operations as `Layer::forward` performs for that output.
    fn policy_subset(&self, nz: &[(u32, f32)], slots: &[u32], out: &mut Vec<f32>) {
        let inputs = self.policy.inputs;
        assert!(nz.iter().all(|&(i, _)| (i as usize) < inputs), "input index out of range");
        assert!(slots.iter().all(|&s| s == u32::MAX || (s as usize) < self.policy.outputs), "action slot out of range");
        out.clear();
        out.resize(slots.len(), 0.0);
        let rows = self.policy_rows.as_ptr();
        for (chunk, out) in slots.chunks(LANES).zip(out.chunks_mut(LANES)) {
            // Lanes without a slot (none left, or no slot) follow slot 0 and
            // are overwritten below.
            let mut off = [0usize; LANES];
            let mut bias = [0f32; LANES];
            for (k, &s) in chunk.iter().enumerate() {
                let s = if s == u32::MAX { 0 } else { s as usize };
                off[k] = s * inputs;
                bias[k] = self.policy.bias[s];
            }
            // SAFETY: every row offset is a valid output's and every index
            // < inputs (asserted above), so each `rows` read is in bounds.
            unsafe {
                let mut acc = simd::load(bias.as_ptr());
                for &(i, xi) in nz {
                    let i = i as usize;
                    let w: [u16; LANES] = std::array::from_fn(|k| *rows.add(off[k] + i));
                    acc = simd::fma(acc, simd::splat(xi), simd::load_f16(w.as_ptr()));
                }
                let mut lanes = [0f32; LANES];
                simd::store(lanes.as_mut_ptr(), acc);
                for (k, (o, &s)) in out.iter_mut().zip(chunk).enumerate() {
                    *o = if s == u32::MAX { f32::NEG_INFINITY } else { lanes[k] };
                }
            }
        }
    }
}
