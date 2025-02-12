#![allow(non_snake_case)]
#![allow(clippy::too_many_lines)]
use ndarray::{prelude::*, ScalarOperand, Zip};
use num_traits::{Float, One, Zero};
use std::{
    fmt::Debug,
    iter::Sum,
    ops::{AddAssign, Mul, Neg, Sub},
};

pub trait Inverse<T: Float> {
    fn det(&self) -> T;

    fn inv(&self) -> Option<Self>
    where
        Self: Sized;

    fn inv_diag(&self) -> Option<Self>
    where
        Self: Sized;

    fn lu_inv(&self) -> Option<Self>
    where
        Self: Sized;

    fn inv_lt(&self) -> Option<Self>
    where
        Self: Sized;

    fn inv_ut(&self) -> Option<Self>
    where
        Self: Sized;

    #[allow(clippy::return_self_not_must_use)]
    fn cholesky(&self) -> Self
    where
        Self: Sized;

    fn lu(&self) -> Option<(Self, Self, Self)>
    where
        Self: Sized;
}

impl<T> Inverse<T> for Array2<T>
where
    T: Float + Debug + ScalarOperand + Sum<T> + AddAssign,
{
    fn det(&self) -> T {
        let s = self.raw_dim();
        assert!(s[0] == s[1]);
        // Flatten to Vec!
        let vm: Vec<T> = self.iter().copied().collect();
        determinant(&vm, s[0])
    }

    fn inv(&self) -> Option<Self> {
        fn submat<T: Float>(res: &mut [T], m: &Array2<T>, sr: usize, sc: usize) {
            let s = m.raw_dim();
            let mut i: usize = 0;
            (0..s[0]).for_each(|r| {
                (0..s[1]).for_each(|c| {
                    if r != sr && c != sc {
                        res[i] = m[(r, c)];
                        i += 1;
                    }
                });
            });
        }

        let s = self.raw_dim();
        assert!(s[0] == s[1]);
        let l = s[0];
        // check if determinant for shortcut cases only
        // Note: Cases must match with following main match
        let det = match l {
            2..=4 => {
                let d = self.det();

                if d.is_zero() {
                    return None;
                }
                d
            }
            _ => T::zero(),
        };
        match l {
            1 => Some(array![[self[(0, 0)].recip()]]),
            2 => Some(array![
                [self[(1, 1)] / det, -self[(0, 1)] / det],
                [-self[(1, 0)] / det, self[(0, 0)] / det],
            ]),
            3 => {
                let m00 = self[(0, 0)];
                let m01 = self[(0, 1)];
                let m02 = self[(0, 2)];
                let m10 = self[(1, 0)];
                let m11 = self[(1, 1)];
                let m12 = self[(1, 2)];
                let m20 = self[(2, 0)];
                let m21 = self[(2, 1)];
                let m22 = self[(2, 2)];

                let x00 = m11 * m22 - m21 * m12;
                let x01 = m02 * m21 - m01 * m22;
                let x02 = m01 * m12 - m02 * m11;
                let x10 = m12 * m20 - m10 * m22;
                let x11 = m00 * m22 - m02 * m20;
                let x12 = m10 * m02 - m00 * m12;
                let x20 = m10 * m21 - m20 * m11;
                let x21 = m20 * m01 - m00 * m21;
                let x22 = m00 * m11 - m10 * m01;

                Some(array![
                    [x00 / det, x01 / det, x02 / det],
                    [x10 / det, x11 / det, x12 / det],
                    [x20 / det, x21 / det, x22 / det]
                ])
            }
            4 => {
                let m00 = self[(0, 0)];
                let m01 = self[(0, 1)];
                let m02 = self[(0, 2)];
                let m03 = self[(0, 3)];
                let m10 = self[(1, 0)];
                let m11 = self[(1, 1)];
                let m12 = self[(1, 2)];
                let m13 = self[(1, 3)];
                let m20 = self[(2, 0)];
                let m21 = self[(2, 1)];
                let m22 = self[(2, 2)];
                let m23 = self[(2, 3)];
                let m30 = self[(3, 0)];
                let m31 = self[(3, 1)];
                let m32 = self[(3, 2)];
                let m33 = self[(3, 3)];

                let x00 = m11 * m22 * m33 - m11 * m32 * m23 - m12 * m21 * m33 + m12 * m31 * m23 + m13 * m21 * m32 - m13 * m31 * m22;

                let x10 = -m10 * m22 * m33 + m10 * m32 * m23 + m12 * m20 * m33 - m12 * m30 * m23 - m13 * m20 * m32 + m13 * m30 * m22;

                let x20 = m10 * m21 * m33 - m10 * m31 * m23 - m11 * m20 * m33 + m11 * m30 * m23 + m13 * m20 * m31 - m13 * m30 * m21;

                let x30 = -m10 * m21 * m32 + m10 * m31 * m22 + m11 * m20 * m32 - m11 * m30 * m22 - m12 * m20 * m31 + m12 * m30 * m21;

                let x01 = -m01 * m22 * m33 + m01 * m32 * m23 + m02 * m21 * m33 - m02 * m31 * m23 - m03 * m21 * m32 + m03 * m31 * m22;

                let x11 = m00 * m22 * m33 - m00 * m32 * m23 - m02 * m20 * m33 + m02 * m30 * m23 + m03 * m20 * m32 - m03 * m30 * m22;

                let x21 = -m00 * m21 * m33 + m00 * m31 * m23 + m01 * m20 * m33 - m01 * m30 * m23 - m03 * m20 * m31 + m03 * m30 * m21;

                let x31 = m00 * m21 * m32 - m00 * m31 * m22 - m01 * m20 * m32 + m01 * m30 * m22 + m02 * m20 * m31 - m02 * m30 * m21;

                let x02 = m01 * m12 * m33 - m01 * m32 * m13 - m02 * m11 * m33 + m02 * m31 * m13 + m03 * m11 * m32 - m03 * m31 * m12;

                let x12 = -m00 * m12 * m33 + m00 * m32 * m13 + m02 * m10 * m33 - m02 * m30 * m13 - m03 * m10 * m32 + m03 * m30 * m12;

                let x22 = m00 * m11 * m33 - m00 * m31 * m13 - m01 * m10 * m33 + m01 * m30 * m13 + m03 * m10 * m31 - m03 * m30 * m11;

                let x03 = -m01 * m12 * m23 + m01 * m22 * m13 + m02 * m11 * m23 - m02 * m21 * m13 - m03 * m11 * m22 + m03 * m21 * m12;

                let x32 = -m00 * m11 * m32 + m00 * m31 * m12 + m01 * m10 * m32 - m01 * m30 * m12 - m02 * m10 * m31 + m02 * m30 * m11;

                let x13 = m00 * m12 * m23 - m00 * m22 * m13 - m02 * m10 * m23 + m02 * m20 * m13 + m03 * m10 * m22 - m03 * m20 * m12;

                let x23 = -m00 * m11 * m23 + m00 * m21 * m13 + m01 * m10 * m23 - m01 * m20 * m13 - m03 * m10 * m21 + m03 * m20 * m11;

                let x33 = m00 * m11 * m22 - m00 * m21 * m12 - m01 * m10 * m22 + m01 * m20 * m12 + m02 * m10 * m21 - m02 * m20 * m11;

                Some(array![
                    [x00 / det, x01 / det, x02 / det, x03 / det],
                    [x10 / det, x11 / det, x12 / det, x13 / det],
                    [x20 / det, x21 / det, x22 / det, x23 / det],
                    [x30 / det, x31 / det, x32 / det, x33 / det]
                ])
            }
            _ => {
                if let Some(res) = self.lu_inv() {
                    Some(res)
                } else {
                    let det = self.det();
                    // Fully expanding any more is too clunky!
                    if det.is_zero() {
                        None
                    } else {
                        let mut cofactors: Array2<T> = Array2::zeros((l, l));
                        let mut res: Vec<T> = vec![T::zero(); (l - 1) * (l - 1)];
                        for r in 0..l {
                            for c in 0..l {
                                // Find submatrix
                                submat(&mut res, self, r, c);
                                let d = determinant(&res, l - 1);
                                cofactors[(c, r)] = if ((r + c) % 2) == 0 { d } else { -d };
                            }
                        }
                        Some(cofactors / det)
                    }
                }
            }
        }
    }

    fn inv_diag(&self) -> Option<Self> {
        let s = self.raw_dim();
        assert!(s[0] == s[1]);

        let mut res: Array2<T> = Array2::zeros(s);

        for i in 0..s[0] {
            res[(i, i)] = self[(i, i)].recip();
        }

        Some(res)
    }

    fn cholesky(&self) -> Self {
        let s = self.raw_dim();
        assert!(s[0] == s[1]);
        let l = s[0];
        let mut res: Array2<T> = Array2::zeros(s);
        for i in 0..l {
            for j in 0..=i {
                let mut s = T::zero();
                for k in 0..j {
                    if !res[(i, k)].is_zero() && !res[(j, k)].is_zero() {
                        s += res[(i, k)] * res[(j, k)];
                    }
                }

                res[(i, j)] = if i == j {
                    (self[(i, i)] - s).sqrt()
                } else {
                    res[(j, j)].recip() * (self[(i, j)] - s)
                };
            }
        }

        res
    }

    fn lu(&self) -> Option<(Self, Self, Self)> {
        fn pivot<T: Float>(A: &Array2<T>) -> Array2<T> {
            let n = A.raw_dim()[0];
            let mut P: Array2<T> = Array::eye(n);

            for (idx, col) in A.axis_iter(Axis(1)).enumerate() {
                // find index of maximum value in column i
                let mut mp = idx;
                for i in idx..n {
                    if col[mp].abs() < col[i].abs() {
                        mp = i;
                    }
                }
                // swap rows when different
                if mp != idx {
                    let (l, r) = P.multi_slice_mut((s![idx, ..], s![mp, ..]));

                    Zip::from(l).and(r).for_each(std::mem::swap);
                }
            }

            P
        }

        let d = self.raw_dim();
        let n = d[0];
        assert_eq!(n, d[1], "LU decomposition must take a square matrix.");

        let P = pivot(self);
        let pA = P.dot(self);

        let mut L: Array2<T> = Array::eye(n);
        let mut U: Array2<T> = Array::zeros((n, n));

        for c in 0..n {
            for r in 0..n {
                let s = U.slice(s![0..r, c]).dot(&L.slice(s![r, 0..r]));

                if s.is_nan() || s.is_infinite() {
                    return None;
                }

                let pAs = pA[[r, c]] - s;

                if r < c + 1 {
                    // U
                    U[[r, c]] = pAs;
                } else {
                    // L
                    L[[r, c]] = (pAs) / U[[c, c]];
                }
            }
        }

        Some((L, U, P))
    }

    fn lu_inv(&self) -> Option<Self> {
        let d = self.raw_dim();
        let n = d[0];

        assert!(d[0] == d[1]);

        if let Some((l, u, p)) = self.lu() {
            let lt = linv(&l, n);
            let ut = uinv(&u, n);

            Some(ut.dot(&lt).dot(&p))
        } else {
            None
        }
    }

    fn inv_lt(&self) -> Option<Self> {
        let n = check_diag(self);

        if n == 0 {
            None
        } else {
            Some(linv(self, n))
        }
    }

    fn inv_ut(&self) -> Option<Self> {
        let n = check_diag(self);

        if n == 0 {
            None
        } else {
            Some(uinv(self, n))
        }
    }
}

fn determinant<T>(vm: &[T], l: usize) -> T
where
    T: Copy + Debug + Zero + One + Mul + PartialEq + Sub<Output = T> + Neg<Output = T> + Sum<T>,
{
    // Must be square matrix l x l!
    // Fully expanding the first few determinants makes a big
    // difference, especially as the generic algorithm is recursive
    match l {
        1 => vm[0],
        2 => vm[0] * vm[3] - vm[2] * vm[1],
        3 => {
            let m00 = vm[0];
            let m01 = vm[1];
            let m02 = vm[2];
            let m10 = vm[3];
            let m11 = vm[4];
            let m12 = vm[5];
            let m20 = vm[6];
            let m21 = vm[7];
            let m22 = vm[8];

            m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20)
        }
        4 => {
            let m00 = vm[0];
            let m01 = vm[1];
            let m02 = vm[2];
            let m03 = vm[3];
            let m10 = vm[4];
            let m11 = vm[5];
            let m12 = vm[6];
            let m13 = vm[7];
            let m20 = vm[8];
            let m21 = vm[9];
            let m22 = vm[10];
            let m23 = vm[11];
            let m30 = vm[12];
            let m31 = vm[13];
            let m32 = vm[14];
            let m33 = vm[15];

            // Note: Compiler optimisation will remove redundant multiplies
            m00 * m21 * m32 * m13 - m00 * m21 * m33 * m12 - m00 * m22 * m31 * m13 + m00 * m22 * m33 * m11 + m00 * m23 * m31 * m12
                - m00 * m23 * m32 * m11
                - m21 * m30 * m02 * m13
                + m21 * m30 * m03 * m12
                - m21 * m32 * m03 * m10
                + m21 * m33 * m02 * m10
                + m22 * m30 * m01 * m13
                - m22 * m30 * m03 * m11
                + m22 * m31 * m03 * m10
                - m22 * m33 * m01 * m10
                - m23 * m30 * m01 * m12
                + m23 * m30 * m02 * m11
                - m23 * m31 * m02 * m10
                + m23 * m32 * m01 * m10
                + m31 * m02 * m13 * m20
                - m31 * m03 * m12 * m20
                - m32 * m01 * m13 * m20
                + m32 * m03 * m11 * m20
                + m33 * m01 * m12 * m20
                - m33 * m02 * m11 * m20
        }
        5 => {
            let m00 = vm[0];
            let m01 = vm[1];
            let m02 = vm[2];
            let m03 = vm[3];
            let m04 = vm[4];
            let m10 = vm[5];
            let m11 = vm[6];
            let m12 = vm[7];
            let m13 = vm[8];
            let m14 = vm[9];
            let m20 = vm[10];
            let m21 = vm[11];
            let m22 = vm[12];
            let m23 = vm[13];
            let m24 = vm[14];
            let m30 = vm[15];
            let m31 = vm[16];
            let m32 = vm[17];
            let m33 = vm[18];
            let m34 = vm[19];
            let m40 = vm[20];
            let m41 = vm[21];
            let m42 = vm[22];
            let m43 = vm[23];
            let m44 = vm[24];

            // Note: Compiler optimisation will remove redundant multiplies
            m00 * m11 * m22 * m33 * m44 - m00 * m11 * m22 * m34 * m43 - m00 * m11 * m23 * m32 * m44
                + m00 * m11 * m23 * m34 * m42
                + m00 * m11 * m24 * m32 * m43
                - m00 * m11 * m24 * m33 * m42
                - m00 * m12 * m21 * m33 * m44
                + m00 * m12 * m21 * m34 * m43
                + m00 * m12 * m23 * m31 * m44
                - m00 * m12 * m23 * m34 * m41
                - m00 * m12 * m24 * m31 * m43
                + m00 * m12 * m24 * m33 * m41
                + m00 * m13 * m21 * m32 * m44
                - m00 * m13 * m21 * m34 * m42
                - m00 * m13 * m22 * m31 * m44
                + m00 * m13 * m22 * m34 * m41
                + m00 * m13 * m24 * m31 * m42
                - m00 * m13 * m24 * m32 * m41
                - m00 * m14 * m21 * m32 * m43
                + m00 * m14 * m21 * m33 * m42
                + m00 * m14 * m22 * m31 * m43
                - m00 * m14 * m22 * m33 * m41
                - m00 * m14 * m23 * m31 * m42
                + m00 * m14 * m23 * m32 * m41
                - m01 * m10 * m22 * m33 * m44
                + m01 * m10 * m22 * m34 * m43
                + m01 * m10 * m23 * m32 * m44
                - m01 * m10 * m23 * m34 * m42
                - m01 * m10 * m24 * m32 * m43
                + m01 * m10 * m24 * m33 * m42
                + m01 * m12 * m20 * m33 * m44
                - m01 * m12 * m20 * m34 * m43
                - m01 * m12 * m23 * m30 * m44
                + m01 * m12 * m23 * m34 * m40
                + m01 * m12 * m24 * m30 * m43
                - m01 * m12 * m24 * m33 * m40
                - m01 * m13 * m20 * m32 * m44
                + m01 * m13 * m20 * m34 * m42
                + m01 * m13 * m22 * m30 * m44
                - m01 * m13 * m22 * m34 * m40
                - m01 * m13 * m24 * m30 * m42
                + m01 * m13 * m24 * m32 * m40
                + m01 * m14 * m20 * m32 * m43
                - m01 * m14 * m20 * m33 * m42
                - m01 * m14 * m22 * m30 * m43
                + m01 * m14 * m22 * m33 * m40
                + m01 * m14 * m23 * m30 * m42
                - m01 * m14 * m23 * m32 * m40
                + m02 * m10 * m21 * m33 * m44
                - m02 * m10 * m21 * m34 * m43
                - m02 * m10 * m23 * m31 * m44
                + m02 * m10 * m23 * m34 * m41
                + m02 * m10 * m24 * m31 * m43
                - m02 * m10 * m24 * m33 * m41
                - m02 * m11 * m20 * m33 * m44
                + m02 * m11 * m20 * m34 * m43
                + m02 * m11 * m23 * m30 * m44
                - m02 * m11 * m23 * m34 * m40
                - m02 * m11 * m24 * m30 * m43
                + m02 * m11 * m24 * m33 * m40
                + m02 * m13 * m20 * m31 * m44
                - m02 * m13 * m20 * m34 * m41
                - m02 * m13 * m21 * m30 * m44
                + m02 * m13 * m21 * m34 * m40
                + m02 * m13 * m24 * m30 * m41
                - m02 * m13 * m24 * m31 * m40
                - m02 * m14 * m20 * m31 * m43
                + m02 * m14 * m20 * m33 * m41
                + m02 * m14 * m21 * m30 * m43
                - m02 * m14 * m21 * m33 * m40
                - m02 * m14 * m23 * m30 * m41
                + m02 * m14 * m23 * m31 * m40
                - m03 * m10 * m21 * m32 * m44
                + m03 * m10 * m21 * m34 * m42
                + m03 * m10 * m22 * m31 * m44
                - m03 * m10 * m22 * m34 * m41
                - m03 * m10 * m24 * m31 * m42
                + m03 * m10 * m24 * m32 * m41
                + m03 * m11 * m20 * m32 * m44
                - m03 * m11 * m20 * m34 * m42
                - m03 * m11 * m22 * m30 * m44
                + m03 * m11 * m22 * m34 * m40
                + m03 * m11 * m24 * m30 * m42
                - m03 * m11 * m24 * m32 * m40
                - m03 * m12 * m20 * m31 * m44
                + m03 * m12 * m20 * m34 * m41
                + m03 * m12 * m21 * m30 * m44
                - m03 * m12 * m21 * m34 * m40
                - m03 * m12 * m24 * m30 * m41
                + m03 * m12 * m24 * m31 * m40
                + m03 * m14 * m20 * m31 * m42
                - m03 * m14 * m20 * m32 * m41
                - m03 * m14 * m21 * m30 * m42
                + m03 * m14 * m21 * m32 * m40
                + m03 * m14 * m22 * m30 * m41
                - m03 * m14 * m22 * m31 * m40
                + m04 * m10 * m21 * m32 * m43
                - m04 * m10 * m21 * m33 * m42
                - m04 * m10 * m22 * m31 * m43
                + m04 * m10 * m22 * m33 * m41
                + m04 * m10 * m23 * m31 * m42
                - m04 * m10 * m23 * m32 * m41
                - m04 * m11 * m20 * m32 * m43
                + m04 * m11 * m20 * m33 * m42
                + m04 * m11 * m22 * m30 * m43
                - m04 * m11 * m22 * m33 * m40
                - m04 * m11 * m23 * m30 * m42
                + m04 * m11 * m23 * m32 * m40
                + m04 * m12 * m20 * m31 * m43
                - m04 * m12 * m20 * m33 * m41
                - m04 * m12 * m21 * m30 * m43
                + m04 * m12 * m21 * m33 * m40
                + m04 * m12 * m23 * m30 * m41
                - m04 * m12 * m23 * m31 * m40
                - m04 * m13 * m20 * m31 * m42
                + m04 * m13 * m20 * m32 * m41
                + m04 * m13 * m21 * m30 * m42
                - m04 * m13 * m21 * m32 * m40
                - m04 * m13 * m22 * m30 * m41
                + m04 * m13 * m22 * m31 * m40
        }
        6 => {
            // Okay, getting very unweldy but worth it
            let m00 = vm[0];
            let m01 = vm[1];
            let m02 = vm[2];
            let m03 = vm[3];
            let m04 = vm[4];
            let m05 = vm[5];
            let m10 = vm[6];
            let m11 = vm[7];
            let m12 = vm[8];
            let m13 = vm[9];
            let m14 = vm[10];
            let m15 = vm[11];
            let m20 = vm[12];
            let m21 = vm[13];
            let m22 = vm[14];
            let m23 = vm[15];
            let m24 = vm[16];
            let m25 = vm[17];
            let m30 = vm[18];
            let m31 = vm[19];
            let m32 = vm[20];
            let m33 = vm[21];
            let m34 = vm[22];
            let m35 = vm[23];
            let m40 = vm[24];
            let m41 = vm[25];
            let m42 = vm[26];
            let m43 = vm[27];
            let m44 = vm[28];
            let m45 = vm[29];
            let m50 = vm[30];
            let m51 = vm[31];
            let m52 = vm[32];
            let m53 = vm[33];
            let m54 = vm[34];
            let m55 = vm[35];

            // Note: Compiler optimisation will remove redundant multiplies
            m00 * m11 * m22 * m33 * m44 * m55 - m00 * m11 * m22 * m33 * m45 * m54 - m00 * m11 * m22 * m34 * m43 * m55
                + m00 * m11 * m22 * m34 * m45 * m53
                + m00 * m11 * m22 * m35 * m43 * m54
                - m00 * m11 * m22 * m35 * m44 * m53
                - m00 * m11 * m23 * m32 * m44 * m55
                + m00 * m11 * m23 * m32 * m45 * m54
                + m00 * m11 * m23 * m34 * m42 * m55
                - m00 * m11 * m23 * m34 * m45 * m52
                - m00 * m11 * m23 * m35 * m42 * m54
                + m00 * m11 * m23 * m35 * m44 * m52
                + m00 * m11 * m24 * m32 * m43 * m55
                - m00 * m11 * m24 * m32 * m45 * m53
                - m00 * m11 * m24 * m33 * m42 * m55
                + m00 * m11 * m24 * m33 * m45 * m52
                + m00 * m11 * m24 * m35 * m42 * m53
                - m00 * m11 * m24 * m35 * m43 * m52
                - m00 * m11 * m25 * m32 * m43 * m54
                + m00 * m11 * m25 * m32 * m44 * m53
                + m00 * m11 * m25 * m33 * m42 * m54
                - m00 * m11 * m25 * m33 * m44 * m52
                - m00 * m11 * m25 * m34 * m42 * m53
                + m00 * m11 * m25 * m34 * m43 * m52
                - m00 * m12 * m21 * m33 * m44 * m55
                + m00 * m12 * m21 * m33 * m45 * m54
                + m00 * m12 * m21 * m34 * m43 * m55
                - m00 * m12 * m21 * m34 * m45 * m53
                - m00 * m12 * m21 * m35 * m43 * m54
                + m00 * m12 * m21 * m35 * m44 * m53
                + m00 * m12 * m23 * m31 * m44 * m55
                - m00 * m12 * m23 * m31 * m45 * m54
                - m00 * m12 * m23 * m34 * m41 * m55
                + m00 * m12 * m23 * m34 * m45 * m51
                + m00 * m12 * m23 * m35 * m41 * m54
                - m00 * m12 * m23 * m35 * m44 * m51
                - m00 * m12 * m24 * m31 * m43 * m55
                + m00 * m12 * m24 * m31 * m45 * m53
                + m00 * m12 * m24 * m33 * m41 * m55
                - m00 * m12 * m24 * m33 * m45 * m51
                - m00 * m12 * m24 * m35 * m41 * m53
                + m00 * m12 * m24 * m35 * m43 * m51
                + m00 * m12 * m25 * m31 * m43 * m54
                - m00 * m12 * m25 * m31 * m44 * m53
                - m00 * m12 * m25 * m33 * m41 * m54
                + m00 * m12 * m25 * m33 * m44 * m51
                + m00 * m12 * m25 * m34 * m41 * m53
                - m00 * m12 * m25 * m34 * m43 * m51
                + m00 * m13 * m21 * m32 * m44 * m55
                - m00 * m13 * m21 * m32 * m45 * m54
                - m00 * m13 * m21 * m34 * m42 * m55
                + m00 * m13 * m21 * m34 * m45 * m52
                + m00 * m13 * m21 * m35 * m42 * m54
                - m00 * m13 * m21 * m35 * m44 * m52
                - m00 * m13 * m22 * m31 * m44 * m55
                + m00 * m13 * m22 * m31 * m45 * m54
                + m00 * m13 * m22 * m34 * m41 * m55
                - m00 * m13 * m22 * m34 * m45 * m51
                - m00 * m13 * m22 * m35 * m41 * m54
                + m00 * m13 * m22 * m35 * m44 * m51
                + m00 * m13 * m24 * m31 * m42 * m55
                - m00 * m13 * m24 * m31 * m45 * m52
                - m00 * m13 * m24 * m32 * m41 * m55
                + m00 * m13 * m24 * m32 * m45 * m51
                + m00 * m13 * m24 * m35 * m41 * m52
                - m00 * m13 * m24 * m35 * m42 * m51
                - m00 * m13 * m25 * m31 * m42 * m54
                + m00 * m13 * m25 * m31 * m44 * m52
                + m00 * m13 * m25 * m32 * m41 * m54
                - m00 * m13 * m25 * m32 * m44 * m51
                - m00 * m13 * m25 * m34 * m41 * m52
                + m00 * m13 * m25 * m34 * m42 * m51
                - m00 * m14 * m21 * m32 * m43 * m55
                + m00 * m14 * m21 * m32 * m45 * m53
                + m00 * m14 * m21 * m33 * m42 * m55
                - m00 * m14 * m21 * m33 * m45 * m52
                - m00 * m14 * m21 * m35 * m42 * m53
                + m00 * m14 * m21 * m35 * m43 * m52
                + m00 * m14 * m22 * m31 * m43 * m55
                - m00 * m14 * m22 * m31 * m45 * m53
                - m00 * m14 * m22 * m33 * m41 * m55
                + m00 * m14 * m22 * m33 * m45 * m51
                + m00 * m14 * m22 * m35 * m41 * m53
                - m00 * m14 * m22 * m35 * m43 * m51
                - m00 * m14 * m23 * m31 * m42 * m55
                + m00 * m14 * m23 * m31 * m45 * m52
                + m00 * m14 * m23 * m32 * m41 * m55
                - m00 * m14 * m23 * m32 * m45 * m51
                - m00 * m14 * m23 * m35 * m41 * m52
                + m00 * m14 * m23 * m35 * m42 * m51
                + m00 * m14 * m25 * m31 * m42 * m53
                - m00 * m14 * m25 * m31 * m43 * m52
                - m00 * m14 * m25 * m32 * m41 * m53
                + m00 * m14 * m25 * m32 * m43 * m51
                + m00 * m14 * m25 * m33 * m41 * m52
                - m00 * m14 * m25 * m33 * m42 * m51
                + m00 * m15 * m21 * m32 * m43 * m54
                - m00 * m15 * m21 * m32 * m44 * m53
                - m00 * m15 * m21 * m33 * m42 * m54
                + m00 * m15 * m21 * m33 * m44 * m52
                + m00 * m15 * m21 * m34 * m42 * m53
                - m00 * m15 * m21 * m34 * m43 * m52
                - m00 * m15 * m22 * m31 * m43 * m54
                + m00 * m15 * m22 * m31 * m44 * m53
                + m00 * m15 * m22 * m33 * m41 * m54
                - m00 * m15 * m22 * m33 * m44 * m51
                - m00 * m15 * m22 * m34 * m41 * m53
                + m00 * m15 * m22 * m34 * m43 * m51
                + m00 * m15 * m23 * m31 * m42 * m54
                - m00 * m15 * m23 * m31 * m44 * m52
                - m00 * m15 * m23 * m32 * m41 * m54
                + m00 * m15 * m23 * m32 * m44 * m51
                + m00 * m15 * m23 * m34 * m41 * m52
                - m00 * m15 * m23 * m34 * m42 * m51
                - m00 * m15 * m24 * m31 * m42 * m53
                + m00 * m15 * m24 * m31 * m43 * m52
                + m00 * m15 * m24 * m32 * m41 * m53
                - m00 * m15 * m24 * m32 * m43 * m51
                - m00 * m15 * m24 * m33 * m41 * m52
                + m00 * m15 * m24 * m33 * m42 * m51
                - m01 * m10 * m22 * m33 * m44 * m55
                + m01 * m10 * m22 * m33 * m45 * m54
                + m01 * m10 * m22 * m34 * m43 * m55
                - m01 * m10 * m22 * m34 * m45 * m53
                - m01 * m10 * m22 * m35 * m43 * m54
                + m01 * m10 * m22 * m35 * m44 * m53
                + m01 * m10 * m23 * m32 * m44 * m55
                - m01 * m10 * m23 * m32 * m45 * m54
                - m01 * m10 * m23 * m34 * m42 * m55
                + m01 * m10 * m23 * m34 * m45 * m52
                + m01 * m10 * m23 * m35 * m42 * m54
                - m01 * m10 * m23 * m35 * m44 * m52
                - m01 * m10 * m24 * m32 * m43 * m55
                + m01 * m10 * m24 * m32 * m45 * m53
                + m01 * m10 * m24 * m33 * m42 * m55
                - m01 * m10 * m24 * m33 * m45 * m52
                - m01 * m10 * m24 * m35 * m42 * m53
                + m01 * m10 * m24 * m35 * m43 * m52
                + m01 * m10 * m25 * m32 * m43 * m54
                - m01 * m10 * m25 * m32 * m44 * m53
                - m01 * m10 * m25 * m33 * m42 * m54
                + m01 * m10 * m25 * m33 * m44 * m52
                + m01 * m10 * m25 * m34 * m42 * m53
                - m01 * m10 * m25 * m34 * m43 * m52
                + m01 * m12 * m20 * m33 * m44 * m55
                - m01 * m12 * m20 * m33 * m45 * m54
                - m01 * m12 * m20 * m34 * m43 * m55
                + m01 * m12 * m20 * m34 * m45 * m53
                + m01 * m12 * m20 * m35 * m43 * m54
                - m01 * m12 * m20 * m35 * m44 * m53
                - m01 * m12 * m23 * m30 * m44 * m55
                + m01 * m12 * m23 * m30 * m45 * m54
                + m01 * m12 * m23 * m34 * m40 * m55
                - m01 * m12 * m23 * m34 * m45 * m50
                - m01 * m12 * m23 * m35 * m40 * m54
                + m01 * m12 * m23 * m35 * m44 * m50
                + m01 * m12 * m24 * m30 * m43 * m55
                - m01 * m12 * m24 * m30 * m45 * m53
                - m01 * m12 * m24 * m33 * m40 * m55
                + m01 * m12 * m24 * m33 * m45 * m50
                + m01 * m12 * m24 * m35 * m40 * m53
                - m01 * m12 * m24 * m35 * m43 * m50
                - m01 * m12 * m25 * m30 * m43 * m54
                + m01 * m12 * m25 * m30 * m44 * m53
                + m01 * m12 * m25 * m33 * m40 * m54
                - m01 * m12 * m25 * m33 * m44 * m50
                - m01 * m12 * m25 * m34 * m40 * m53
                + m01 * m12 * m25 * m34 * m43 * m50
                - m01 * m13 * m20 * m32 * m44 * m55
                + m01 * m13 * m20 * m32 * m45 * m54
                + m01 * m13 * m20 * m34 * m42 * m55
                - m01 * m13 * m20 * m34 * m45 * m52
                - m01 * m13 * m20 * m35 * m42 * m54
                + m01 * m13 * m20 * m35 * m44 * m52
                + m01 * m13 * m22 * m30 * m44 * m55
                - m01 * m13 * m22 * m30 * m45 * m54
                - m01 * m13 * m22 * m34 * m40 * m55
                + m01 * m13 * m22 * m34 * m45 * m50
                + m01 * m13 * m22 * m35 * m40 * m54
                - m01 * m13 * m22 * m35 * m44 * m50
                - m01 * m13 * m24 * m30 * m42 * m55
                + m01 * m13 * m24 * m30 * m45 * m52
                + m01 * m13 * m24 * m32 * m40 * m55
                - m01 * m13 * m24 * m32 * m45 * m50
                - m01 * m13 * m24 * m35 * m40 * m52
                + m01 * m13 * m24 * m35 * m42 * m50
                + m01 * m13 * m25 * m30 * m42 * m54
                - m01 * m13 * m25 * m30 * m44 * m52
                - m01 * m13 * m25 * m32 * m40 * m54
                + m01 * m13 * m25 * m32 * m44 * m50
                + m01 * m13 * m25 * m34 * m40 * m52
                - m01 * m13 * m25 * m34 * m42 * m50
                + m01 * m14 * m20 * m32 * m43 * m55
                - m01 * m14 * m20 * m32 * m45 * m53
                - m01 * m14 * m20 * m33 * m42 * m55
                + m01 * m14 * m20 * m33 * m45 * m52
                + m01 * m14 * m20 * m35 * m42 * m53
                - m01 * m14 * m20 * m35 * m43 * m52
                - m01 * m14 * m22 * m30 * m43 * m55
                + m01 * m14 * m22 * m30 * m45 * m53
                + m01 * m14 * m22 * m33 * m40 * m55
                - m01 * m14 * m22 * m33 * m45 * m50
                - m01 * m14 * m22 * m35 * m40 * m53
                + m01 * m14 * m22 * m35 * m43 * m50
                + m01 * m14 * m23 * m30 * m42 * m55
                - m01 * m14 * m23 * m30 * m45 * m52
                - m01 * m14 * m23 * m32 * m40 * m55
                + m01 * m14 * m23 * m32 * m45 * m50
                + m01 * m14 * m23 * m35 * m40 * m52
                - m01 * m14 * m23 * m35 * m42 * m50
                - m01 * m14 * m25 * m30 * m42 * m53
                + m01 * m14 * m25 * m30 * m43 * m52
                + m01 * m14 * m25 * m32 * m40 * m53
                - m01 * m14 * m25 * m32 * m43 * m50
                - m01 * m14 * m25 * m33 * m40 * m52
                + m01 * m14 * m25 * m33 * m42 * m50
                - m01 * m15 * m20 * m32 * m43 * m54
                + m01 * m15 * m20 * m32 * m44 * m53
                + m01 * m15 * m20 * m33 * m42 * m54
                - m01 * m15 * m20 * m33 * m44 * m52
                - m01 * m15 * m20 * m34 * m42 * m53
                + m01 * m15 * m20 * m34 * m43 * m52
                + m01 * m15 * m22 * m30 * m43 * m54
                - m01 * m15 * m22 * m30 * m44 * m53
                - m01 * m15 * m22 * m33 * m40 * m54
                + m01 * m15 * m22 * m33 * m44 * m50
                + m01 * m15 * m22 * m34 * m40 * m53
                - m01 * m15 * m22 * m34 * m43 * m50
                - m01 * m15 * m23 * m30 * m42 * m54
                + m01 * m15 * m23 * m30 * m44 * m52
                + m01 * m15 * m23 * m32 * m40 * m54
                - m01 * m15 * m23 * m32 * m44 * m50
                - m01 * m15 * m23 * m34 * m40 * m52
                + m01 * m15 * m23 * m34 * m42 * m50
                + m01 * m15 * m24 * m30 * m42 * m53
                - m01 * m15 * m24 * m30 * m43 * m52
                - m01 * m15 * m24 * m32 * m40 * m53
                + m01 * m15 * m24 * m32 * m43 * m50
                + m01 * m15 * m24 * m33 * m40 * m52
                - m01 * m15 * m24 * m33 * m42 * m50
                + m02 * m10 * m21 * m33 * m44 * m55
                - m02 * m10 * m21 * m33 * m45 * m54
                - m02 * m10 * m21 * m34 * m43 * m55
                + m02 * m10 * m21 * m34 * m45 * m53
                + m02 * m10 * m21 * m35 * m43 * m54
                - m02 * m10 * m21 * m35 * m44 * m53
                - m02 * m10 * m23 * m31 * m44 * m55
                + m02 * m10 * m23 * m31 * m45 * m54
                + m02 * m10 * m23 * m34 * m41 * m55
                - m02 * m10 * m23 * m34 * m45 * m51
                - m02 * m10 * m23 * m35 * m41 * m54
                + m02 * m10 * m23 * m35 * m44 * m51
                + m02 * m10 * m24 * m31 * m43 * m55
                - m02 * m10 * m24 * m31 * m45 * m53
                - m02 * m10 * m24 * m33 * m41 * m55
                + m02 * m10 * m24 * m33 * m45 * m51
                + m02 * m10 * m24 * m35 * m41 * m53
                - m02 * m10 * m24 * m35 * m43 * m51
                - m02 * m10 * m25 * m31 * m43 * m54
                + m02 * m10 * m25 * m31 * m44 * m53
                + m02 * m10 * m25 * m33 * m41 * m54
                - m02 * m10 * m25 * m33 * m44 * m51
                - m02 * m10 * m25 * m34 * m41 * m53
                + m02 * m10 * m25 * m34 * m43 * m51
                - m02 * m11 * m20 * m33 * m44 * m55
                + m02 * m11 * m20 * m33 * m45 * m54
                + m02 * m11 * m20 * m34 * m43 * m55
                - m02 * m11 * m20 * m34 * m45 * m53
                - m02 * m11 * m20 * m35 * m43 * m54
                + m02 * m11 * m20 * m35 * m44 * m53
                + m02 * m11 * m23 * m30 * m44 * m55
                - m02 * m11 * m23 * m30 * m45 * m54
                - m02 * m11 * m23 * m34 * m40 * m55
                + m02 * m11 * m23 * m34 * m45 * m50
                + m02 * m11 * m23 * m35 * m40 * m54
                - m02 * m11 * m23 * m35 * m44 * m50
                - m02 * m11 * m24 * m30 * m43 * m55
                + m02 * m11 * m24 * m30 * m45 * m53
                + m02 * m11 * m24 * m33 * m40 * m55
                - m02 * m11 * m24 * m33 * m45 * m50
                - m02 * m11 * m24 * m35 * m40 * m53
                + m02 * m11 * m24 * m35 * m43 * m50
                + m02 * m11 * m25 * m30 * m43 * m54
                - m02 * m11 * m25 * m30 * m44 * m53
                - m02 * m11 * m25 * m33 * m40 * m54
                + m02 * m11 * m25 * m33 * m44 * m50
                + m02 * m11 * m25 * m34 * m40 * m53
                - m02 * m11 * m25 * m34 * m43 * m50
                + m02 * m13 * m20 * m31 * m44 * m55
                - m02 * m13 * m20 * m31 * m45 * m54
                - m02 * m13 * m20 * m34 * m41 * m55
                + m02 * m13 * m20 * m34 * m45 * m51
                + m02 * m13 * m20 * m35 * m41 * m54
                - m02 * m13 * m20 * m35 * m44 * m51
                - m02 * m13 * m21 * m30 * m44 * m55
                + m02 * m13 * m21 * m30 * m45 * m54
                + m02 * m13 * m21 * m34 * m40 * m55
                - m02 * m13 * m21 * m34 * m45 * m50
                - m02 * m13 * m21 * m35 * m40 * m54
                + m02 * m13 * m21 * m35 * m44 * m50
                + m02 * m13 * m24 * m30 * m41 * m55
                - m02 * m13 * m24 * m30 * m45 * m51
                - m02 * m13 * m24 * m31 * m40 * m55
                + m02 * m13 * m24 * m31 * m45 * m50
                + m02 * m13 * m24 * m35 * m40 * m51
                - m02 * m13 * m24 * m35 * m41 * m50
                - m02 * m13 * m25 * m30 * m41 * m54
                + m02 * m13 * m25 * m30 * m44 * m51
                + m02 * m13 * m25 * m31 * m40 * m54
                - m02 * m13 * m25 * m31 * m44 * m50
                - m02 * m13 * m25 * m34 * m40 * m51
                + m02 * m13 * m25 * m34 * m41 * m50
                - m02 * m14 * m20 * m31 * m43 * m55
                + m02 * m14 * m20 * m31 * m45 * m53
                + m02 * m14 * m20 * m33 * m41 * m55
                - m02 * m14 * m20 * m33 * m45 * m51
                - m02 * m14 * m20 * m35 * m41 * m53
                + m02 * m14 * m20 * m35 * m43 * m51
                + m02 * m14 * m21 * m30 * m43 * m55
                - m02 * m14 * m21 * m30 * m45 * m53
                - m02 * m14 * m21 * m33 * m40 * m55
                + m02 * m14 * m21 * m33 * m45 * m50
                + m02 * m14 * m21 * m35 * m40 * m53
                - m02 * m14 * m21 * m35 * m43 * m50
                - m02 * m14 * m23 * m30 * m41 * m55
                + m02 * m14 * m23 * m30 * m45 * m51
                + m02 * m14 * m23 * m31 * m40 * m55
                - m02 * m14 * m23 * m31 * m45 * m50
                - m02 * m14 * m23 * m35 * m40 * m51
                + m02 * m14 * m23 * m35 * m41 * m50
                + m02 * m14 * m25 * m30 * m41 * m53
                - m02 * m14 * m25 * m30 * m43 * m51
                - m02 * m14 * m25 * m31 * m40 * m53
                + m02 * m14 * m25 * m31 * m43 * m50
                + m02 * m14 * m25 * m33 * m40 * m51
                - m02 * m14 * m25 * m33 * m41 * m50
                + m02 * m15 * m20 * m31 * m43 * m54
                - m02 * m15 * m20 * m31 * m44 * m53
                - m02 * m15 * m20 * m33 * m41 * m54
                + m02 * m15 * m20 * m33 * m44 * m51
                + m02 * m15 * m20 * m34 * m41 * m53
                - m02 * m15 * m20 * m34 * m43 * m51
                - m02 * m15 * m21 * m30 * m43 * m54
                + m02 * m15 * m21 * m30 * m44 * m53
                + m02 * m15 * m21 * m33 * m40 * m54
                - m02 * m15 * m21 * m33 * m44 * m50
                - m02 * m15 * m21 * m34 * m40 * m53
                + m02 * m15 * m21 * m34 * m43 * m50
                + m02 * m15 * m23 * m30 * m41 * m54
                - m02 * m15 * m23 * m30 * m44 * m51
                - m02 * m15 * m23 * m31 * m40 * m54
                + m02 * m15 * m23 * m31 * m44 * m50
                + m02 * m15 * m23 * m34 * m40 * m51
                - m02 * m15 * m23 * m34 * m41 * m50
                - m02 * m15 * m24 * m30 * m41 * m53
                + m02 * m15 * m24 * m30 * m43 * m51
                + m02 * m15 * m24 * m31 * m40 * m53
                - m02 * m15 * m24 * m31 * m43 * m50
                - m02 * m15 * m24 * m33 * m40 * m51
                + m02 * m15 * m24 * m33 * m41 * m50
                - m03 * m10 * m21 * m32 * m44 * m55
                + m03 * m10 * m21 * m32 * m45 * m54
                + m03 * m10 * m21 * m34 * m42 * m55
                - m03 * m10 * m21 * m34 * m45 * m52
                - m03 * m10 * m21 * m35 * m42 * m54
                + m03 * m10 * m21 * m35 * m44 * m52
                + m03 * m10 * m22 * m31 * m44 * m55
                - m03 * m10 * m22 * m31 * m45 * m54
                - m03 * m10 * m22 * m34 * m41 * m55
                + m03 * m10 * m22 * m34 * m45 * m51
                + m03 * m10 * m22 * m35 * m41 * m54
                - m03 * m10 * m22 * m35 * m44 * m51
                - m03 * m10 * m24 * m31 * m42 * m55
                + m03 * m10 * m24 * m31 * m45 * m52
                + m03 * m10 * m24 * m32 * m41 * m55
                - m03 * m10 * m24 * m32 * m45 * m51
                - m03 * m10 * m24 * m35 * m41 * m52
                + m03 * m10 * m24 * m35 * m42 * m51
                + m03 * m10 * m25 * m31 * m42 * m54
                - m03 * m10 * m25 * m31 * m44 * m52
                - m03 * m10 * m25 * m32 * m41 * m54
                + m03 * m10 * m25 * m32 * m44 * m51
                + m03 * m10 * m25 * m34 * m41 * m52
                - m03 * m10 * m25 * m34 * m42 * m51
                + m03 * m11 * m20 * m32 * m44 * m55
                - m03 * m11 * m20 * m32 * m45 * m54
                - m03 * m11 * m20 * m34 * m42 * m55
                + m03 * m11 * m20 * m34 * m45 * m52
                + m03 * m11 * m20 * m35 * m42 * m54
                - m03 * m11 * m20 * m35 * m44 * m52
                - m03 * m11 * m22 * m30 * m44 * m55
                + m03 * m11 * m22 * m30 * m45 * m54
                + m03 * m11 * m22 * m34 * m40 * m55
                - m03 * m11 * m22 * m34 * m45 * m50
                - m03 * m11 * m22 * m35 * m40 * m54
                + m03 * m11 * m22 * m35 * m44 * m50
                + m03 * m11 * m24 * m30 * m42 * m55
                - m03 * m11 * m24 * m30 * m45 * m52
                - m03 * m11 * m24 * m32 * m40 * m55
                + m03 * m11 * m24 * m32 * m45 * m50
                + m03 * m11 * m24 * m35 * m40 * m52
                - m03 * m11 * m24 * m35 * m42 * m50
                - m03 * m11 * m25 * m30 * m42 * m54
                + m03 * m11 * m25 * m30 * m44 * m52
                + m03 * m11 * m25 * m32 * m40 * m54
                - m03 * m11 * m25 * m32 * m44 * m50
                - m03 * m11 * m25 * m34 * m40 * m52
                + m03 * m11 * m25 * m34 * m42 * m50
                - m03 * m12 * m20 * m31 * m44 * m55
                + m03 * m12 * m20 * m31 * m45 * m54
                + m03 * m12 * m20 * m34 * m41 * m55
                - m03 * m12 * m20 * m34 * m45 * m51
                - m03 * m12 * m20 * m35 * m41 * m54
                + m03 * m12 * m20 * m35 * m44 * m51
                + m03 * m12 * m21 * m30 * m44 * m55
                - m03 * m12 * m21 * m30 * m45 * m54
                - m03 * m12 * m21 * m34 * m40 * m55
                + m03 * m12 * m21 * m34 * m45 * m50
                + m03 * m12 * m21 * m35 * m40 * m54
                - m03 * m12 * m21 * m35 * m44 * m50
                - m03 * m12 * m24 * m30 * m41 * m55
                + m03 * m12 * m24 * m30 * m45 * m51
                + m03 * m12 * m24 * m31 * m40 * m55
                - m03 * m12 * m24 * m31 * m45 * m50
                - m03 * m12 * m24 * m35 * m40 * m51
                + m03 * m12 * m24 * m35 * m41 * m50
                + m03 * m12 * m25 * m30 * m41 * m54
                - m03 * m12 * m25 * m30 * m44 * m51
                - m03 * m12 * m25 * m31 * m40 * m54
                + m03 * m12 * m25 * m31 * m44 * m50
                + m03 * m12 * m25 * m34 * m40 * m51
                - m03 * m12 * m25 * m34 * m41 * m50
                + m03 * m14 * m20 * m31 * m42 * m55
                - m03 * m14 * m20 * m31 * m45 * m52
                - m03 * m14 * m20 * m32 * m41 * m55
                + m03 * m14 * m20 * m32 * m45 * m51
                + m03 * m14 * m20 * m35 * m41 * m52
                - m03 * m14 * m20 * m35 * m42 * m51
                - m03 * m14 * m21 * m30 * m42 * m55
                + m03 * m14 * m21 * m30 * m45 * m52
                + m03 * m14 * m21 * m32 * m40 * m55
                - m03 * m14 * m21 * m32 * m45 * m50
                - m03 * m14 * m21 * m35 * m40 * m52
                + m03 * m14 * m21 * m35 * m42 * m50
                + m03 * m14 * m22 * m30 * m41 * m55
                - m03 * m14 * m22 * m30 * m45 * m51
                - m03 * m14 * m22 * m31 * m40 * m55
                + m03 * m14 * m22 * m31 * m45 * m50
                + m03 * m14 * m22 * m35 * m40 * m51
                - m03 * m14 * m22 * m35 * m41 * m50
                - m03 * m14 * m25 * m30 * m41 * m52
                + m03 * m14 * m25 * m30 * m42 * m51
                + m03 * m14 * m25 * m31 * m40 * m52
                - m03 * m14 * m25 * m31 * m42 * m50
                - m03 * m14 * m25 * m32 * m40 * m51
                + m03 * m14 * m25 * m32 * m41 * m50
                - m03 * m15 * m20 * m31 * m42 * m54
                + m03 * m15 * m20 * m31 * m44 * m52
                + m03 * m15 * m20 * m32 * m41 * m54
                - m03 * m15 * m20 * m32 * m44 * m51
                - m03 * m15 * m20 * m34 * m41 * m52
                + m03 * m15 * m20 * m34 * m42 * m51
                + m03 * m15 * m21 * m30 * m42 * m54
                - m03 * m15 * m21 * m30 * m44 * m52
                - m03 * m15 * m21 * m32 * m40 * m54
                + m03 * m15 * m21 * m32 * m44 * m50
                + m03 * m15 * m21 * m34 * m40 * m52
                - m03 * m15 * m21 * m34 * m42 * m50
                - m03 * m15 * m22 * m30 * m41 * m54
                + m03 * m15 * m22 * m30 * m44 * m51
                + m03 * m15 * m22 * m31 * m40 * m54
                - m03 * m15 * m22 * m31 * m44 * m50
                - m03 * m15 * m22 * m34 * m40 * m51
                + m03 * m15 * m22 * m34 * m41 * m50
                + m03 * m15 * m24 * m30 * m41 * m52
                - m03 * m15 * m24 * m30 * m42 * m51
                - m03 * m15 * m24 * m31 * m40 * m52
                + m03 * m15 * m24 * m31 * m42 * m50
                + m03 * m15 * m24 * m32 * m40 * m51
                - m03 * m15 * m24 * m32 * m41 * m50
                + m04 * m10 * m21 * m32 * m43 * m55
                - m04 * m10 * m21 * m32 * m45 * m53
                - m04 * m10 * m21 * m33 * m42 * m55
                + m04 * m10 * m21 * m33 * m45 * m52
                + m04 * m10 * m21 * m35 * m42 * m53
                - m04 * m10 * m21 * m35 * m43 * m52
                - m04 * m10 * m22 * m31 * m43 * m55
                + m04 * m10 * m22 * m31 * m45 * m53
                + m04 * m10 * m22 * m33 * m41 * m55
                - m04 * m10 * m22 * m33 * m45 * m51
                - m04 * m10 * m22 * m35 * m41 * m53
                + m04 * m10 * m22 * m35 * m43 * m51
                + m04 * m10 * m23 * m31 * m42 * m55
                - m04 * m10 * m23 * m31 * m45 * m52
                - m04 * m10 * m23 * m32 * m41 * m55
                + m04 * m10 * m23 * m32 * m45 * m51
                + m04 * m10 * m23 * m35 * m41 * m52
                - m04 * m10 * m23 * m35 * m42 * m51
                - m04 * m10 * m25 * m31 * m42 * m53
                + m04 * m10 * m25 * m31 * m43 * m52
                + m04 * m10 * m25 * m32 * m41 * m53
                - m04 * m10 * m25 * m32 * m43 * m51
                - m04 * m10 * m25 * m33 * m41 * m52
                + m04 * m10 * m25 * m33 * m42 * m51
                - m04 * m11 * m20 * m32 * m43 * m55
                + m04 * m11 * m20 * m32 * m45 * m53
                + m04 * m11 * m20 * m33 * m42 * m55
                - m04 * m11 * m20 * m33 * m45 * m52
                - m04 * m11 * m20 * m35 * m42 * m53
                + m04 * m11 * m20 * m35 * m43 * m52
                + m04 * m11 * m22 * m30 * m43 * m55
                - m04 * m11 * m22 * m30 * m45 * m53
                - m04 * m11 * m22 * m33 * m40 * m55
                + m04 * m11 * m22 * m33 * m45 * m50
                + m04 * m11 * m22 * m35 * m40 * m53
                - m04 * m11 * m22 * m35 * m43 * m50
                - m04 * m11 * m23 * m30 * m42 * m55
                + m04 * m11 * m23 * m30 * m45 * m52
                + m04 * m11 * m23 * m32 * m40 * m55
                - m04 * m11 * m23 * m32 * m45 * m50
                - m04 * m11 * m23 * m35 * m40 * m52
                + m04 * m11 * m23 * m35 * m42 * m50
                + m04 * m11 * m25 * m30 * m42 * m53
                - m04 * m11 * m25 * m30 * m43 * m52
                - m04 * m11 * m25 * m32 * m40 * m53
                + m04 * m11 * m25 * m32 * m43 * m50
                + m04 * m11 * m25 * m33 * m40 * m52
                - m04 * m11 * m25 * m33 * m42 * m50
                + m04 * m12 * m20 * m31 * m43 * m55
                - m04 * m12 * m20 * m31 * m45 * m53
                - m04 * m12 * m20 * m33 * m41 * m55
                + m04 * m12 * m20 * m33 * m45 * m51
                + m04 * m12 * m20 * m35 * m41 * m53
                - m04 * m12 * m20 * m35 * m43 * m51
                - m04 * m12 * m21 * m30 * m43 * m55
                + m04 * m12 * m21 * m30 * m45 * m53
                + m04 * m12 * m21 * m33 * m40 * m55
                - m04 * m12 * m21 * m33 * m45 * m50
                - m04 * m12 * m21 * m35 * m40 * m53
                + m04 * m12 * m21 * m35 * m43 * m50
                + m04 * m12 * m23 * m30 * m41 * m55
                - m04 * m12 * m23 * m30 * m45 * m51
                - m04 * m12 * m23 * m31 * m40 * m55
                + m04 * m12 * m23 * m31 * m45 * m50
                + m04 * m12 * m23 * m35 * m40 * m51
                - m04 * m12 * m23 * m35 * m41 * m50
                - m04 * m12 * m25 * m30 * m41 * m53
                + m04 * m12 * m25 * m30 * m43 * m51
                + m04 * m12 * m25 * m31 * m40 * m53
                - m04 * m12 * m25 * m31 * m43 * m50
                - m04 * m12 * m25 * m33 * m40 * m51
                + m04 * m12 * m25 * m33 * m41 * m50
                - m04 * m13 * m20 * m31 * m42 * m55
                + m04 * m13 * m20 * m31 * m45 * m52
                + m04 * m13 * m20 * m32 * m41 * m55
                - m04 * m13 * m20 * m32 * m45 * m51
                - m04 * m13 * m20 * m35 * m41 * m52
                + m04 * m13 * m20 * m35 * m42 * m51
                + m04 * m13 * m21 * m30 * m42 * m55
                - m04 * m13 * m21 * m30 * m45 * m52
                - m04 * m13 * m21 * m32 * m40 * m55
                + m04 * m13 * m21 * m32 * m45 * m50
                + m04 * m13 * m21 * m35 * m40 * m52
                - m04 * m13 * m21 * m35 * m42 * m50
                - m04 * m13 * m22 * m30 * m41 * m55
                + m04 * m13 * m22 * m30 * m45 * m51
                + m04 * m13 * m22 * m31 * m40 * m55
                - m04 * m13 * m22 * m31 * m45 * m50
                - m04 * m13 * m22 * m35 * m40 * m51
                + m04 * m13 * m22 * m35 * m41 * m50
                + m04 * m13 * m25 * m30 * m41 * m52
                - m04 * m13 * m25 * m30 * m42 * m51
                - m04 * m13 * m25 * m31 * m40 * m52
                + m04 * m13 * m25 * m31 * m42 * m50
                + m04 * m13 * m25 * m32 * m40 * m51
                - m04 * m13 * m25 * m32 * m41 * m50
                + m04 * m15 * m20 * m31 * m42 * m53
                - m04 * m15 * m20 * m31 * m43 * m52
                - m04 * m15 * m20 * m32 * m41 * m53
                + m04 * m15 * m20 * m32 * m43 * m51
                + m04 * m15 * m20 * m33 * m41 * m52
                - m04 * m15 * m20 * m33 * m42 * m51
                - m04 * m15 * m21 * m30 * m42 * m53
                + m04 * m15 * m21 * m30 * m43 * m52
                + m04 * m15 * m21 * m32 * m40 * m53
                - m04 * m15 * m21 * m32 * m43 * m50
                - m04 * m15 * m21 * m33 * m40 * m52
                + m04 * m15 * m21 * m33 * m42 * m50
                + m04 * m15 * m22 * m30 * m41 * m53
                - m04 * m15 * m22 * m30 * m43 * m51
                - m04 * m15 * m22 * m31 * m40 * m53
                + m04 * m15 * m22 * m31 * m43 * m50
                + m04 * m15 * m22 * m33 * m40 * m51
                - m04 * m15 * m22 * m33 * m41 * m50
                - m04 * m15 * m23 * m30 * m41 * m52
                + m04 * m15 * m23 * m30 * m42 * m51
                + m04 * m15 * m23 * m31 * m40 * m52
                - m04 * m15 * m23 * m31 * m42 * m50
                - m04 * m15 * m23 * m32 * m40 * m51
                + m04 * m15 * m23 * m32 * m41 * m50
                - m05 * m10 * m21 * m32 * m43 * m54
                + m05 * m10 * m21 * m32 * m44 * m53
                + m05 * m10 * m21 * m33 * m42 * m54
                - m05 * m10 * m21 * m33 * m44 * m52
                - m05 * m10 * m21 * m34 * m42 * m53
                + m05 * m10 * m21 * m34 * m43 * m52
                + m05 * m10 * m22 * m31 * m43 * m54
                - m05 * m10 * m22 * m31 * m44 * m53
                - m05 * m10 * m22 * m33 * m41 * m54
                + m05 * m10 * m22 * m33 * m44 * m51
                + m05 * m10 * m22 * m34 * m41 * m53
                - m05 * m10 * m22 * m34 * m43 * m51
                - m05 * m10 * m23 * m31 * m42 * m54
                + m05 * m10 * m23 * m31 * m44 * m52
                + m05 * m10 * m23 * m32 * m41 * m54
                - m05 * m10 * m23 * m32 * m44 * m51
                - m05 * m10 * m23 * m34 * m41 * m52
                + m05 * m10 * m23 * m34 * m42 * m51
                + m05 * m10 * m24 * m31 * m42 * m53
                - m05 * m10 * m24 * m31 * m43 * m52
                - m05 * m10 * m24 * m32 * m41 * m53
                + m05 * m10 * m24 * m32 * m43 * m51
                + m05 * m10 * m24 * m33 * m41 * m52
                - m05 * m10 * m24 * m33 * m42 * m51
                + m05 * m11 * m20 * m32 * m43 * m54
                - m05 * m11 * m20 * m32 * m44 * m53
                - m05 * m11 * m20 * m33 * m42 * m54
                + m05 * m11 * m20 * m33 * m44 * m52
                + m05 * m11 * m20 * m34 * m42 * m53
                - m05 * m11 * m20 * m34 * m43 * m52
                - m05 * m11 * m22 * m30 * m43 * m54
                + m05 * m11 * m22 * m30 * m44 * m53
                + m05 * m11 * m22 * m33 * m40 * m54
                - m05 * m11 * m22 * m33 * m44 * m50
                - m05 * m11 * m22 * m34 * m40 * m53
                + m05 * m11 * m22 * m34 * m43 * m50
                + m05 * m11 * m23 * m30 * m42 * m54
                - m05 * m11 * m23 * m30 * m44 * m52
                - m05 * m11 * m23 * m32 * m40 * m54
                + m05 * m11 * m23 * m32 * m44 * m50
                + m05 * m11 * m23 * m34 * m40 * m52
                - m05 * m11 * m23 * m34 * m42 * m50
                - m05 * m11 * m24 * m30 * m42 * m53
                + m05 * m11 * m24 * m30 * m43 * m52
                + m05 * m11 * m24 * m32 * m40 * m53
                - m05 * m11 * m24 * m32 * m43 * m50
                - m05 * m11 * m24 * m33 * m40 * m52
                + m05 * m11 * m24 * m33 * m42 * m50
                - m05 * m12 * m20 * m31 * m43 * m54
                + m05 * m12 * m20 * m31 * m44 * m53
                + m05 * m12 * m20 * m33 * m41 * m54
                - m05 * m12 * m20 * m33 * m44 * m51
                - m05 * m12 * m20 * m34 * m41 * m53
                + m05 * m12 * m20 * m34 * m43 * m51
                + m05 * m12 * m21 * m30 * m43 * m54
                - m05 * m12 * m21 * m30 * m44 * m53
                - m05 * m12 * m21 * m33 * m40 * m54
                + m05 * m12 * m21 * m33 * m44 * m50
                + m05 * m12 * m21 * m34 * m40 * m53
                - m05 * m12 * m21 * m34 * m43 * m50
                - m05 * m12 * m23 * m30 * m41 * m54
                + m05 * m12 * m23 * m30 * m44 * m51
                + m05 * m12 * m23 * m31 * m40 * m54
                - m05 * m12 * m23 * m31 * m44 * m50
                - m05 * m12 * m23 * m34 * m40 * m51
                + m05 * m12 * m23 * m34 * m41 * m50
                + m05 * m12 * m24 * m30 * m41 * m53
                - m05 * m12 * m24 * m30 * m43 * m51
                - m05 * m12 * m24 * m31 * m40 * m53
                + m05 * m12 * m24 * m31 * m43 * m50
                + m05 * m12 * m24 * m33 * m40 * m51
                - m05 * m12 * m24 * m33 * m41 * m50
                + m05 * m13 * m20 * m31 * m42 * m54
                - m05 * m13 * m20 * m31 * m44 * m52
                - m05 * m13 * m20 * m32 * m41 * m54
                + m05 * m13 * m20 * m32 * m44 * m51
                + m05 * m13 * m20 * m34 * m41 * m52
                - m05 * m13 * m20 * m34 * m42 * m51
                - m05 * m13 * m21 * m30 * m42 * m54
                + m05 * m13 * m21 * m30 * m44 * m52
                + m05 * m13 * m21 * m32 * m40 * m54
                - m05 * m13 * m21 * m32 * m44 * m50
                - m05 * m13 * m21 * m34 * m40 * m52
                + m05 * m13 * m21 * m34 * m42 * m50
                + m05 * m13 * m22 * m30 * m41 * m54
                - m05 * m13 * m22 * m30 * m44 * m51
                - m05 * m13 * m22 * m31 * m40 * m54
                + m05 * m13 * m22 * m31 * m44 * m50
                + m05 * m13 * m22 * m34 * m40 * m51
                - m05 * m13 * m22 * m34 * m41 * m50
                - m05 * m13 * m24 * m30 * m41 * m52
                + m05 * m13 * m24 * m30 * m42 * m51
                + m05 * m13 * m24 * m31 * m40 * m52
                - m05 * m13 * m24 * m31 * m42 * m50
                - m05 * m13 * m24 * m32 * m40 * m51
                + m05 * m13 * m24 * m32 * m41 * m50
                - m05 * m14 * m20 * m31 * m42 * m53
                + m05 * m14 * m20 * m31 * m43 * m52
                + m05 * m14 * m20 * m32 * m41 * m53
                - m05 * m14 * m20 * m32 * m43 * m51
                - m05 * m14 * m20 * m33 * m41 * m52
                + m05 * m14 * m20 * m33 * m42 * m51
                + m05 * m14 * m21 * m30 * m42 * m53
                - m05 * m14 * m21 * m30 * m43 * m52
                - m05 * m14 * m21 * m32 * m40 * m53
                + m05 * m14 * m21 * m32 * m43 * m50
                + m05 * m14 * m21 * m33 * m40 * m52
                - m05 * m14 * m21 * m33 * m42 * m50
                - m05 * m14 * m22 * m30 * m41 * m53
                + m05 * m14 * m22 * m30 * m43 * m51
                + m05 * m14 * m22 * m31 * m40 * m53
                - m05 * m14 * m22 * m31 * m43 * m50
                - m05 * m14 * m22 * m33 * m40 * m51
                + m05 * m14 * m22 * m33 * m41 * m50
                + m05 * m14 * m23 * m30 * m41 * m52
                - m05 * m14 * m23 * m30 * m42 * m51
                - m05 * m14 * m23 * m31 * m40 * m52
                + m05 * m14 * m23 * m31 * m42 * m50
                + m05 * m14 * m23 * m32 * m40 * m51
                - m05 * m14 * m23 * m32 * m41 * m50
        }
        _ =>
        // Now do it the traditional way
        {
            // Reuseable result to reduce memory allocations
            let l1 = l - 1;
            let mut res: Vec<T> = vm.to_vec(); // same shape/size as vm
                                               // This is n^3
            (0..l)
                .map(|i| {
                    // Makes a modest difference
                    if vm[i].is_zero() {
                        T::zero()
                    } else {
                        for r in 0..l1 {
                            let d = r * l1;
                            let s = (r + 1) * l;
                            for c in 0..l1 {
                                res[d + c] = vm[s + c + usize::from(c >= i)];
                            }
                        }
                        let v = vm[i] * determinant(&res, l1);
                        if (i % 2) == 0 {
                            v
                        } else {
                            -v
                        }
                    }
                })
                .sum::<T>()
        }
    }
}

fn check_diag<T: Float>(m: &Array2<T>) -> usize {
    let d = m.raw_dim();
    assert!(d[0] == d[1]);

    for i in 0..d[0] {
        if m[(i, i)].is_zero() {
            return 0;
        }
    }

    d[0]
}

fn linv<T: Float>(l: &Array2<T>, n: usize) -> Array2<T> {
    let mut m: Array2<T> = Array2::zeros((n, n));

    for i in 0..n {
        m[(i, i)] = l[(i, i)].recip();

        for j in 0..i {
            for k in j..i {
                m[(i, j)] = m[(i, j)] + l[(i, k)] * m[(k, j)];
            }

            m[(i, j)] = -m[(i, j)] * m[(i, i)];
        }
    }

    m
}

fn uinv<T: Float>(u: &Array2<T>, n: usize) -> Array2<T> {
    linv(&u.t().to_owned(), n).t().to_owned()
}

// #[cfg(test)]
// mod test_inverse {
//     use crate::Inverse;
//     use ndarray::prelude::*;
//     use num_traits::Float;
//     use std::fmt::Debug;

//     const EPS: f64 = 1e-8;

//     fn to_vec<T: Float>(m: &Array2<T>) -> Vec<T> {
//         m.iter().map(|&e| e).collect()
//     }

//     /// utility function to compare vectors of Floats
//     fn compare_vecs<T: Float + Debug>(v1: &Array2<T>, v2: &Array2<T>,
// epsilon: T) -> bool {         to_vec(v1)
//             .into_iter()
//             .zip(to_vec(v2))
//             .map(|(a, b)| Float::abs(a.abs() - b.abs()) < epsilon)
//             .all(|b| b)
//     }

//     #[test]
//     fn test_1x1() {
//         let a: Array2<f64> = array![[2.0]];
//         let inv = a.inv().expect("Sods Law");
//         let back = inv.inv();
//         assert!(compare_vecs(&a, &back.unwrap(), EPS));
//         let expected = array![[0.5]];
//         assert!(compare_vecs(&inv, &expected, EPS));
//     }

//     #[test]
//     fn test_2x2() {
//         let a: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
//         let inv = a.inv().expect("Sods Law");
//         let back = inv.inv();
//         assert!(compare_vecs(&a, &back.unwrap(), EPS));
//         let expected = array![[-2.0, 1.0], [1.5, -0.5]];
//         assert!(compare_vecs(&inv, &expected, EPS));
//     }

//     #[test]
//     fn test_3x3() {
//         let a: Array2<f64> = array![[1.0, 0.0, 3.0], [2.0, 1.0, 6.0], [1.0,
// 0.0, 9.0]];         let inv = a.inv().expect("Sods Law");
//         let back = inv.inv();
//         assert!(compare_vecs(&a, &back.unwrap(), EPS));
//         let expected = array![[1.5, 0.0, -0.5], [-2.0, 1.0, -0.0],
// [-0.16666667, 0.0, 0.16666667]];         assert!(compare_vecs(&inv,
// &expected, EPS));     }

//     #[test]
//     fn test_4x4() {
//         let a: Array2<f64> = array![
//             [-68.0, 68.0, -16.0, 4.0],
//             [-36.0, 35.0, -9.0, 3.0],
//             [48.0, -47.0, 11.0, -3.0],
//             [64.0, -64.0, 16.0, -4.0]
//         ];
//         let inv = a.inv().expect("Sods Law");
//         let back = inv.inv();
//         eprintln!("{:?}", to_vec(&inv));
//         // This fails with LU Inverse only
//         assert!(compare_vecs(&a, &back.unwrap(), EPS));
//         let expected = array![[1.25, 0.5, 1.5, 0.5], [1.5, 0.5, 1.5, 0.75],
// [1.5, 0.5, 0.5, 1.5], [2.0, 2.0, 2.0, 1.75]];         assert!(compare_vecs(&
// inv, &expected, EPS));     }

//     #[test]
//     fn test_5x5() {
//         let a: Array2<f64> = array![
//             [10.0, 1.0, 7.0, 1.0, 5.0],
//             [2.0, 4.0, 8.0, 3.0, 2.0],
//             [5.0, 1.0, 2.0, 9.0, 10.0],
//             [6.0, 9.0, 9.0, 7.0, 3.0],
//             [1.0, 8.0, 8.0, 10.0, 5.0],
//         ];
//         let inv = a.inv().expect("Sods Law");
//         let back = inv.inv().unwrap();
//         // some concern here, precision might not be ideal (try 1e-5)
//         // Note: Precision is better for LU Inverse and full EPS can be used
//         assert!(compare_vecs(&a, &back, EPS));
//         let expected: Array2<f64> = array![
//             [-11.98, 15.64, 9.32, 10.34, -19.12],
//             [33.62, -44.16, -26.08, -28.46, 53.28],
//             [-9.36, 12.48, 7.24, 7.88, -14.84],
//             [-37.2, 48.6, 28.8, 31.6, -58.8],
//             [37.98, -49.64, -29.32, -32.34, 60.12]
//         ];
//         assert!(compare_vecs(&inv, &expected, EPS));
//     }

//     #[test]
//     fn test_5x5b() {
//         let a: Array2<f64> = array![
//             [8.12658, 1.02214, 2.67754, 7.00746, 0.16869],
//             [3.31581, 0.63792, 4.28436, 5.61229, 1.19785],
//             [8.13028, 6.48437, 6.91838, 2.68860, 6.60971],
//             [2.85578, 4.40362, 1.13265, 8.32311, 4.41366],
//             [7.93379, 4.09682, 4.86444, 6.67282, 3.57098]
//         ];
//         let inv = a.inv().expect("Sods Law");
//         let back = inv.inv().unwrap();
//         // Precision is good here!
//         assert!(compare_vecs(&a, &back, EPS));
//         let expected: Array2<f64> = array![
//             [0.50074759, -0.02510542, 0.33877844, 0.09275141, -0.7569348],
//             [-1.58848174, -0.64356647, -1.32943575, -0.57849057, 3.46664012],
//             [-0.49917931, 0.12378294, -0.30452856, -0.25789124, 0.86447499],
//             [-0.0513424, 0.06348965, -0.1164256, 0.0585164, 0.12430141],
//             [1.48578933, 0.50685455, 1.40491123, 0.69956389, -3.42524083]
//         ];
//         assert!(compare_vecs(&inv, &expected, EPS));
//     }

//     #[test]
//     fn test_6x6() {
//         let a: Array2<f64> = array![
//             [1.0, 1.0, 3.0, 4.0, 9.0, 3.0],
//             [10.0, 10.0, 1.0, 2.0, 2.0, 5.0],
//             [2.0, 9.0, 6.0, 10.0, 10.0, 9.0],
//             [10.0, 9.0, 9.0, 7.0, 3.0, 6.0],
//             [7.0, 6.0, 6.0, 2.0, 9.0, 5.0],
//             [3.0, 8.0, 1.0, 4.0, 1.0, 5.0]
//         ];
//         let inv = a.inv().expect("Sods Law");
//         eprintln!("{:?}", to_vec(&inv));
//         let back = inv.inv();
//         // This fails with LU Inverse only
//         assert!(compare_vecs(&a, &back.unwrap(), EPS));
//         let expected = array![
//             [-0.53881418, 0.57793399, 0.6653423, -0.08374083, -0.16962103,
// -1.18215159],             [2.16075795, -1.52750611, -2.44070905, 0.44132029,
// 0.32457213, 3.77017115],             [0.21454768, -0.41503667, -0.39425428,
// 0.14792176, 0.19743276, 0.62102689],             [0.70018337, -0.24052567,
// -0.525978, 0.20354523, -0.21179707, 0.73471883],             [0.85055012,
// -0.47157702, -0.82793399, 0.1106357, 0.1146088, 1.20415648],
// [-3.90709046, 2.46699267, 4.17114914, -0.87041565, -0.31051345, -6.07579462]
//         ];
//         assert!(compare_vecs(&inv, &expected, EPS));
//     }

//     #[test]
//     fn test_7x7() {
//         let a: Array2<f64> = array![
//             [4.3552, 6.25851, 4.12662, 1.93708, 0.21272, 3.25683, 6.53326],
//             [4.24746, 1.84137, 6.71904, 0.59754, 3.5806, 3.63597, 5.347],
//             [2.30479, 1.70591, 3.05354, 1.82188, 5.27839, 7.9166, 2.04607],
//             [2.40158, 6.38524, 7.90296, 4.69683, 6.63801, 7.32958, 1.45936],
//             [0.42456, 6.47456, 1.55398, 8.28979, 4.20987, 0.90401, 4.94587],
//             [5.78903, 1.92032, 6.20261, 5.78543, 1.94331, 8.25178, 7.47273],
//             [1.44797, 7.41157, 7.69495, 8.90113, 3.05983, 0.41582, 6.42932]
//         ];
//         let inv = a.inv().expect("Sods Law");
//         let back = inv.inv();
//         assert!(compare_vecs(&a, &back.unwrap(), EPS));
//         let expected = array![
//             [-0.0819729, 0.30947774, -0.90286793, 0.49545542, 0.47316173,
// 0.30651669, -0.71946285],             [0.15235561, -0.07559205, -0.0070691,
// 0.06757486, 0.01026404, -0.08019532, -0.01972633],             [-0.03064041,
// -0.00625996, 0.07810921, -0.01780696, -0.19716224, -0.03302802, 0.20558502],
//             [-0.11358415, -0.02543206, -0.24035635, 0.1260263, 0.1346972,
// 0.16555344, -0.111583],             [-0.1016236, 0.21480081, -0.0698384,
// 0.05521421, 0.20158013, -0.05049775, -0.16205801],             [0.0760218,
// -0.20124886, 0.34308958, -0.12448475, -0.18988991, -0.02734616, 0.18705112],
//             [0.08020185, -0.02906765, 0.46181332, -0.36087425, -0.15255756,
// -0.14045526, 0.31376606]         ];
//         assert!(compare_vecs(&inv, &expected, EPS));
//     }

//     #[cfg(not(debug_assertions))]
//     use std::time::{Duration, Instant};

//     #[test]
//     #[cfg(not(debug_assertions))]
//     // This is very fast when compiled with full optimisations
//     // About 50x slower in debug mode! Thanks Optimizer.
//     fn bench_6x6() {
//         let a: Array2<f64> = array![
//             [1.0, 1.0, 3.0, 4.0, 9.0, 3.0],
//             [10.0, 10.0, 1.0, 2.0, 2.0, 5.0],
//             [2.0, 9.0, 6.0, 10.0, 10.0, 9.0],
//             [10.0, 9.0, 9.0, 7.0, 3.0, 6.0],
//             [7.0, 6.0, 6.0, 2.0, 9.0, 5.0],
//             [3.0, 8.0, 1.0, 4.0, 1.0, 5.0]
//         ];

//         let start = Instant::now();
//         let mut _q = a.inv();
//         for _ in 0..100000 {
//             _q = a.inv();
//         }
//         // Takes about a second on average modern desktop / laptop
//         assert!(start.elapsed() < Duration::new(2, 0));
//     }

//     #[test]
//     fn test_diag() {
//         let a: Array2<f64> = array![
//             [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
//             [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
//             [0.0, 0.0, 6.0, 0.0, 0.0, 0.0],
//             [0.0, 0.0, 0.0, 7.0, 0.0, 0.0],
//             [0.0, 0.0, 0.0, 0.0, 9.0, 0.0],
//             [0.0, 0.0, 0.0, 0.0, 0.0, 5.0]
//         ];
//         let inv = a.inv_diag().expect("Sods Law");
//         let back = inv.inv_diag();
//         assert!(compare_vecs(&a, &back.unwrap(), EPS));
//         let expected = a.inv().expect("Sods Law");
//         assert!(compare_vecs(&inv, &expected, EPS));
//     }

//     #[test]
//     fn test_cholesky() {
//         let a = array![[25.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0,
// 11.0]];

//         let res = a.cholesky();

//         let expected = array![[5.0, 0.0, 0.0], [3.0, 3.0, 0.0], [-1.0, 1.0,
// 3.]];

//         assert!(compare_vecs(&res, &expected, EPS));

//         let a = array![
//             [18.0, 22.0, 54.0, 42.0],
//             [22.0, 70.0, 86.0, 62.0],
//             [54.0, 86.0, 174.0, 134.0],
//             [42.0, 62.0, 134.0, 106.0]
//         ];

//         let res = a.cholesky();

//         let expected = array![
//             [4.242640687119285, 0.0, 0.0, 0.0],
//             [5.185449728701349, 6.565905201197403, 0.0, 0.0],
//             [12.727922061357857, 3.0460384954008553, 1.6497422479090704,
// 0.0],             [9.899494936611667, 1.624553864213788, 1.8497110052313648,
// 1.3926212476456026]         ];

//         assert!(compare_vecs(&res, &expected, EPS));

//         let a = array![[2.0, 0.1], [0.1, 3.0]];

//         let res = a.cholesky().t().to_owned();

//         let expected = array![[1.41421356, 0.07071068], [0.0, 1.73060683]];

//         assert!(compare_vecs(&res, &expected, EPS));

//         let expected_sq = array![[2.005, 0.12237238], [0.12237238, 2.995]];

//         assert!(compare_vecs(&res.dot(&res.t()), &expected_sq, EPS));
//     }

//     #[test]
//     fn test_triangular() {
//         let a = array![
//             [4.242640687119285, 0.0, 0.0, 0.0],
//             [5.185449728701349, 6.565905201197403, 0.0, 0.0],
//             [12.727922061357857, 3.0460384954008553, 1.6497422479090704,
// 0.0],             [9.899494936611667, 1.624553864213788, 1.8497110052313648,
// 1.3926212476456026]         ];

//         assert!(compare_vecs(&a.inv().unwrap(), &a.inv_lt().unwrap(), EPS));
//         assert!(compare_vecs(&a.t().to_owned().inv().unwrap(),
// &a.t().to_owned().inv_ut().unwrap(), EPS));     }
// }
