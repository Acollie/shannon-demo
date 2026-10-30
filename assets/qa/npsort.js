// numpy's default argsort (kind="quicksort") for float arrays, ported so that ties come out in the same
// order as in the Python system: introsort, median-of-3 pivot, insertion sort below 16 elements
// (numpy 2.5: partitions while pr - pl > 15), heapsort past the depth limit.
// Verified against numpy 2.5.4 on random arrays with many ties.

function less(a, b) { return a < b || (b !== b && a === a); }

function aheapsort(v, t, off, n) {
  const A = k => t[off + k - 1];
  const S = (k, x) => { t[off + k - 1] = x; };
  let i, j, l, tmp;
  for (l = n >> 1; l > 0; --l) {
    tmp = A(l);
    for (i = l, j = l << 1; j <= n;) {
      if (j < n && less(v[A(j)], v[A(j + 1)])) j += 1;
      if (less(v[tmp], v[A(j)])) { S(i, A(j)); i = j; j += j; } else break;
    }
    S(i, tmp);
  }
  for (; n > 1;) {
    tmp = A(n); S(n, A(1)); n -= 1;
    for (i = 1, j = 2; j <= n;) {
      if (j < n && less(v[A(j)], v[A(j + 1)])) j++;
      if (less(v[tmp], v[A(j)])) { S(i, A(j)); i = j; j += j; } else break;
    }
    S(i, tmp);
  }
}

export function argsort(v) {
  const num = v.length;
  const t = Array.from({ length: num }, (_, i) => i);
  if (num < 2) return t;
  let pl = 0, pr = num - 1;
  const stack = [], depth = [];
  let cdepth = (31 - Math.clz32(num)) * 2;
  const sw = (a, b) => { const x = t[a]; t[a] = t[b]; t[b] = x; };
  for (;;) {
    if (cdepth < 0) {
      aheapsort(v, t, pl, pr - pl + 1);
    } else {
      while (pr - pl > 15) {
        const pm = pl + ((pr - pl) >> 1);
        if (less(v[t[pm]], v[t[pl]])) sw(pm, pl);
        if (less(v[t[pr]], v[t[pm]])) sw(pr, pm);
        if (less(v[t[pm]], v[t[pl]])) sw(pm, pl);
        const vp = v[t[pm]];
        let pi = pl, pj = pr - 1;
        sw(pm, pj);
        for (;;) {
          do { ++pi; } while (less(v[t[pi]], vp));
          do { --pj; } while (less(vp, v[t[pj]]));
          if (pi >= pj) break;
          sw(pi, pj);
        }
        sw(pi, pr - 1);
        if (pi - pl < pr - pi) { stack.push(pi + 1, pr); pr = pi - 1; }
        else { stack.push(pl, pi - 1); pl = pi + 1; }
        depth.push(--cdepth);
      }
      for (let pi = pl + 1; pi <= pr; ++pi) {
        const vi = t[pi], vp = v[vi];
        let pj = pi, pk = pi - 1;
        while (pj > pl && less(vp, v[t[pk]])) { t[pj--] = t[pk--]; }
        t[pj] = vi;
      }
    }
    if (!stack.length) break;
    pr = stack.pop(); pl = stack.pop(); cdepth = depth.pop();
  }
  return t;
}
