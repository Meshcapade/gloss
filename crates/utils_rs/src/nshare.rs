#![allow(unexpected_cfgs)]
#![cfg(not(doctest))]

/// Converts a 1d type to a ndarray 1d array type.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait ToNdarray1 {
    type Out;

    fn into_ndarray1(self) -> Self::Out;
}

/// Converts a 2d type to a ndarray 2d array type.
///
/// Coordinates are in (row, col).
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait ToNdarray2 {
    type Out;

    fn into_ndarray2(self) -> Self::Out;
}

/// Converts a 3d type to a ndarray 2d array type.
///
/// Coordinates are in `(channel, row, col)`, where channel is typically a color
/// channel, or they are in `(z, y, x)`.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait ToNdarray3 {
    type Out;

    fn into_ndarray3(self) -> Self::Out;
}

/// Borrows a 1d type to a ndarray 1d array type.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait RefNdarray1 {
    type Out;

    fn ref_ndarray1(self) -> Self::Out;
}

/// Borrows a 2d type to a ndarray 2d array type.
///
/// Coordinates are in (row, col).
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait RefNdarray2 {
    type Out;

    fn ref_ndarray2(self) -> Self::Out;
}

/// Borrows a 3d type to a ndarray 2d array type.
///
/// Coordinates are in `(channel, row, col)`, where channel is typically a color
/// channel, or they are in `(z, y, x)`.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait RefNdarray3 {
    type Out;

    fn ref_ndarray3(self) -> Self::Out;
}

/// Mutably borrows a 1d type to a ndarray 1d array type.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait MutNdarray1 {
    type Out;

    fn mut_ndarray1(self) -> Self::Out;
}

/// Mutably borrows a 2d type to a ndarray 2d array type.
///
/// Coordinates are in (row, col).
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait MutNdarray2 {
    type Out;

    fn mut_ndarray2(self) -> Self::Out;
}

/// Mutably borrows a 3d type to a ndarray 2d array type.
///
/// Coordinates are in `(channel, row, col)`, where channel is typically a color
/// channel, or they are in `(z, y, x)`.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait MutNdarray3 {
    type Out;

    fn mut_ndarray3(self) -> Self::Out;
}

/// Converts a 1 or 2 dimensional type to a nalgebra type.
///
/// This uses an associated type to avoid ambiguity for the compiler.
/// By calling this, the compiler always knows the returned type.
pub trait ToNalgebra {
    type Out;

    fn into_nalgebra(self) -> Self::Out;
}

use core::convert::TryFrom;
use nalgebra::Dyn as Dy;

/// ```
/// use nshare::ToNalgebra;
///
/// let arr = ndarray::arr1(&[0.1, 0.2, 0.3, 0.4]);
/// let m = arr.view().into_nalgebra();
/// assert!(m.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
/// assert_eq!(m.shape(), (4, 1));
/// ```
impl<'a, T> ToNalgebra for ndarray::ArrayView1<'a, T>
where
    T: nalgebra::Scalar,
{
    // type Out = nalgebra::DVectorSlice<'a, T>;
    type Out = nalgebra::DVectorView<'a, T>;
    fn into_nalgebra(self) -> Self::Out {
        let len = Dy(self.len());
        let ptr = self.as_ptr();
        let stride: usize = TryFrom::try_from(self.strides()[0]).expect("Negative stride");
        let storage = unsafe { nalgebra::ViewStorage::from_raw_parts(ptr, (len, nalgebra::Const::<1>), (nalgebra::Const::<1>, Dy(stride))) };
        nalgebra::Matrix::from_data(storage)
    }
}
/// ```
/// use nshare::ToNalgebra;
///
/// let mut arr = ndarray::arr1(&[0.1, 0.2, 0.3, 0.4]);
/// let m = arr.view_mut().into_nalgebra();
/// assert!(m.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
/// assert_eq!(m.shape(), (4, 1));
/// ```
#[allow(clippy::drop_non_drop)]
impl<'a, T> ToNalgebra for ndarray::ArrayViewMut1<'a, T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DVectorViewMut<'a, T>;
    fn into_nalgebra(mut self) -> Self::Out {
        let len = Dy(self.len());
        let stride: usize = TryFrom::try_from(self.strides()[0]).expect("Negative stride");
        let ptr = self.as_mut_ptr();
        let storage = unsafe {
            // Drop to not have simultaneously the ndarray and nalgebra valid.
            drop(self);
            nalgebra::ViewStorageMut::from_raw_parts(ptr, (len, nalgebra::Const::<1>), (nalgebra::Const::<1>, Dy(stride)))
        };
        nalgebra::Matrix::from_data(storage)
    }
}

/// ```
/// use nshare::ToNalgebra;
///
/// let arr = ndarray::arr1(&[0.1, 0.2, 0.3, 0.4]);
/// let m = arr.into_nalgebra();
/// assert!(m.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
/// assert_eq!(m.shape(), (4, 1));
/// ```
impl<T> ToNalgebra for ndarray::Array1<T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DVector<T>;
    fn into_nalgebra(self) -> Self::Out {
        let len = Dy(self.len());
        Self::Out::from_vec_generic(len, nalgebra::Const::<1>, self.into_raw_vec_and_offset().0)
    }
}

/// ```
/// use nshare::ToNalgebra;
///
/// let arr = ndarray::arr2(&[
///     [0.1, 0.2, 0.3, 0.4],
///     [0.5, 0.6, 0.7, 0.8],
///     [1.1, 1.2, 1.3, 1.4],
///     [1.5, 1.6, 1.7, 1.8],
/// ]);
/// let m = arr.view().into_nalgebra();
/// assert!(m.row(1).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// assert_eq!(m.shape(), (4, 4));
/// assert!(arr
///     .t()
///     .into_nalgebra()
///     .column(1)
///     .iter()
///     .eq(&[0.5, 0.6, 0.7, 0.8]));
/// ```
impl<'a, T> ToNalgebra for ndarray::ArrayView2<'a, T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DMatrixView<'a, T, Dy, Dy>;
    fn into_nalgebra(self) -> Self::Out {
        let nrows = Dy(self.nrows());
        let ncols = Dy(self.ncols());
        let ptr = self.as_ptr();
        let stride_row: usize = TryFrom::try_from(self.strides()[0]).expect("Negative row stride");
        let stride_col: usize = TryFrom::try_from(self.strides()[1]).expect("Negative column stride");
        let storage = unsafe { nalgebra::ViewStorage::from_raw_parts(ptr, (nrows, ncols), (Dy(stride_row), Dy(stride_col))) };
        nalgebra::Matrix::from_data(storage)
    }
}

/// ```
/// use nshare::ToNalgebra;
///
/// let mut arr = ndarray::arr2(&[
///     [0.1, 0.2, 0.3, 0.4],
///     [0.5, 0.6, 0.7, 0.8],
///     [1.1, 1.2, 1.3, 1.4],
///     [1.5, 1.6, 1.7, 1.8],
/// ]);
/// let m = arr.view_mut().into_nalgebra();
/// assert!(m.row(1).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// assert_eq!(m.shape(), (4, 4));
/// assert!(arr
///     .view_mut()
///     .reversed_axes()
///     .into_nalgebra()
///     .column(1)
///     .iter()
///     .eq(&[0.5, 0.6, 0.7, 0.8]));
/// ```
#[allow(clippy::drop_non_drop)]
impl<'a, T> ToNalgebra for ndarray::ArrayViewMut2<'a, T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DMatrixViewMut<'a, T, Dy, Dy>;
    fn into_nalgebra(mut self) -> Self::Out {
        let nrows = Dy(self.nrows());
        let ncols = Dy(self.ncols());
        let stride_row: usize = TryFrom::try_from(self.strides()[0]).expect("Negative row stride");
        let stride_col: usize = TryFrom::try_from(self.strides()[1]).expect("Negative column stride");
        let ptr = self.as_mut_ptr();
        let storage = unsafe {
            // Drop to not have simultaneously the ndarray and nalgebra valid.
            drop(self);
            nalgebra::ViewStorageMut::from_raw_parts(ptr, (nrows, ncols), (Dy(stride_row), Dy(stride_col)))
        };
        nalgebra::Matrix::from_data(storage)
    }
}

/// ```
/// use nshare::ToNalgebra;
///
/// let mut arr = ndarray::arr2(&[
///     [0.1, 0.2, 0.3, 0.4],
///     [0.5, 0.6, 0.7, 0.8],
///     [1.1, 1.2, 1.3, 1.4],
///     [1.5, 1.6, 1.7, 1.8],
/// ]);
/// let m = arr.clone().into_nalgebra();
/// assert!(m.row(1).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// assert_eq!(m.shape(), (4, 4));
/// assert!(arr
///     .reversed_axes()
///     .into_nalgebra()
///     .column(1)
///     .iter()
///     .eq(&[0.5, 0.6, 0.7, 0.8]));
/// ```
impl<T> ToNalgebra for ndarray::Array2<T>
where
    T: nalgebra::Scalar,
{
    type Out = nalgebra::DMatrix<T>;
    fn into_nalgebra(self) -> Self::Out {
        let std_layout = self.is_standard_layout();
        let nrows = Dy(self.nrows());
        let ncols = Dy(self.ncols());
        // let mut res = Self::Out::from_vec_generic(nrows, ncols, self.into_raw_vec());
        let res = {
            if std_layout {
                let res = Self::Out::from_row_slice(self.nrows(), self.ncols(), self.as_slice().unwrap());
                res
            } else {
                Self::Out::from_vec_generic(nrows, ncols, self.into_raw_vec_and_offset().0)
            }
        };
        // if std_layout {
        //     // This can be expensive, but we have no choice since nalgebra VecStorage
        // is always     // column-based.
        //     res.transpose_mut();
        // }
        res
    }
}

use nalgebra::{
    dimension::U1,
    storage::{Storage, StorageMut},
    Dim, Matrix, Scalar, Vector, ViewStorage, ViewStorageMut,
};
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, ShapeBuilder};

/// ```
/// use nalgebra::Vector4;
/// use ndarray::s;
/// use nshare::RefNdarray1;
///
/// let m = Vector4::new(0.1, 0.2, 0.3, 0.4f32);
/// let arr = m.ref_ndarray1();
/// assert!(arr.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
/// assert_eq!(arr.dim(), 4);
/// ```
impl<'a, N: Scalar, R: Dim, S> RefNdarray1 for &'a Vector<N, R, S>
where
    S: Storage<N, R, U1>,
{
    type Out = ArrayView1<'a, N>;

    fn ref_ndarray1(self) -> Self::Out {
        unsafe { ArrayView1::from_shape_ptr((self.shape().0,).strides((self.strides().0,)), self.as_ptr()) }
    }
}

/// ```
/// use nalgebra::Vector4;
/// use ndarray::s;
/// use nshare::MutNdarray1;
///
/// let mut m = Vector4::new(0.1, 0.2, 0.3, 0.4f32);
/// // Set everything to 0.
/// m.mut_ndarray1().fill(0.0);
/// assert!(m.iter().eq(&[0.0; 4]));
/// ```
impl<'a, N: Scalar, R: Dim, S> MutNdarray1 for &'a mut Vector<N, R, S>
where
    S: StorageMut<N, R, U1>,
{
    type Out = ArrayViewMut1<'a, N>;

    fn mut_ndarray1(self) -> Self::Out {
        unsafe { ArrayViewMut1::from_shape_ptr((self.shape().0,).strides((self.strides().0,)), self.as_ptr().cast_mut()) }
    }
}

/// ```
/// use nalgebra::Vector4;
/// use nshare::ToNdarray1;
///
/// let m = Vector4::new(0.1, 0.2, 0.3, 0.4f32);
/// let arr = m.rows(0, 4).into_ndarray1();
/// assert!(arr.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
/// assert_eq!(arr.dim(), 4);
/// ```
impl<'a, N: Scalar, R: Dim, RStride: Dim, CStride: Dim> ToNdarray1 for Vector<N, R, ViewStorage<'a, N, R, U1, RStride, CStride>> {
    type Out = ArrayView1<'a, N>;

    fn into_ndarray1(self) -> Self::Out {
        unsafe { ArrayView1::from_shape_ptr((self.shape().0,).strides((self.strides().0,)), self.as_ptr()) }
    }
}

/// ```
/// use nalgebra::{dimension::U2, Const, Vector4};
/// use nshare::ToNdarray1;
///
/// let mut m = Vector4::new(0.1, 0.2, 0.3, 0.4);
/// let arr = m
///     .rows_generic_with_step_mut::<Const<2>>(0, Const::<2>, 1)
///     .into_ndarray1()
///     .fill(0.0);
/// assert!(m.iter().eq(&[0.0, 0.2, 0.0, 0.4]));
/// ```
impl<'a, N: Scalar, R: Dim, RStride: Dim, CStride: Dim> ToNdarray1 for Matrix<N, R, U1, ViewStorageMut<'a, N, R, U1, RStride, CStride>> {
    type Out = ArrayViewMut1<'a, N>;

    fn into_ndarray1(self) -> Self::Out {
        unsafe { ArrayViewMut1::from_shape_ptr((self.shape().0,).strides((self.strides().0,)), self.as_ptr().cast_mut()) }
    }
}

/// ```
/// use nalgebra::Matrix4;
/// use ndarray::s;
/// use nshare::RefNdarray2;
///
/// let m = Matrix4::new(
///     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
/// );
/// let arr = m.ref_ndarray2();
/// assert!(arr.slice(s![1, ..]).iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// assert_eq!(arr.dim(), (4, 4));
/// ```
impl<'a, N: Scalar, R: Dim, C: Dim, S> RefNdarray2 for &'a Matrix<N, R, C, S>
where
    S: Storage<N, R, C>,
{
    type Out = ArrayView2<'a, N>;

    fn ref_ndarray2(self) -> Self::Out {
        unsafe { ArrayView2::from_shape_ptr(self.shape().strides(self.strides()), self.as_ptr()) }
    }
}

/// ```
/// use nalgebra::Matrix4;
/// use ndarray::s;
/// use nshare::MutNdarray2;
///
/// let mut m = Matrix4::new(
///     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
/// );
/// let arr = m.mut_ndarray2().slice_mut(s![1, ..]).fill(0.0);
/// assert!(m.row(1).iter().eq(&[0.0; 4]));
/// ```
impl<'a, N: Scalar, R: Dim, C: Dim, S> MutNdarray2 for &'a mut Matrix<N, R, C, S>
where
    S: StorageMut<N, R, C>,
{
    type Out = ArrayViewMut2<'a, N>;

    fn mut_ndarray2(self) -> Self::Out {
        unsafe { ArrayViewMut2::from_shape_ptr(self.shape().strides(self.strides()), self.as_ptr().cast_mut()) }
    }
}

/// ```
/// use nalgebra::Matrix4;
/// use nshare::ToNdarray2;
///
/// let m = Matrix4::new(
///     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
/// );
/// let arr = m.row(1).into_ndarray2();
/// assert!(arr.iter().eq(&[0.5, 0.6, 0.7, 0.8]));
/// assert_eq!(arr.dim(), (1, 4));
/// ```
impl<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim> ToNdarray2 for Matrix<N, R, C, ViewStorage<'a, N, R, C, RStride, CStride>> {
    type Out = ArrayView2<'a, N>;

    fn into_ndarray2(self) -> Self::Out {
        unsafe { ArrayView2::from_shape_ptr(self.shape().strides(self.strides()), self.as_ptr()) }
    }
}

/// ```
/// use nalgebra::Matrix4;
/// use nshare::ToNdarray2;
///
/// let mut m = Matrix4::new(
///     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
/// );
/// let arr = m.row_mut(1).into_ndarray2().fill(0.0);
/// assert!(m.row(1).iter().eq(&[0.0; 4]));
/// ```
impl<'a, N: Scalar, R: Dim, C: Dim, RStride: Dim, CStride: Dim> ToNdarray2 for Matrix<N, R, C, ViewStorageMut<'a, N, R, C, RStride, CStride>> {
    type Out = ArrayViewMut2<'a, N>;

    fn into_ndarray2(self) -> Self::Out {
        unsafe { ArrayViewMut2::from_shape_ptr(self.shape().strides(self.strides()), self.as_ptr().cast_mut()) }
    }
}

#[cfg(feature = "nalgebra_std")]
mod std_impl {
    use super::*;
    use nalgebra::{allocator::Allocator, DVector, DefaultAllocator, Dynamic, VecStorage};
    use ndarray::{Array1, Array2};
    /// ```
    /// use nalgebra::DVector;
    /// use ndarray::s;
    /// use nshare::ToNdarray1;
    ///
    /// let m = DVector::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
    /// let arr = m.into_ndarray1();
    /// assert_eq!(arr.dim(), 4);
    /// assert!(arr.iter().eq(&[0.1, 0.2, 0.3, 0.4]));
    /// ```
    impl<'a, N: Scalar> ToNdarray1 for DVector<N> {
        type Out = Array1<N>;

        fn into_ndarray1(self) -> Self::Out {
            Array1::from_shape_vec((self.shape().0,), self.data.into()).unwrap()
        }
    }

    /// ```
    /// use nalgebra::{
    ///     dimension::{Dynamic, U4},
    ///     Matrix,
    /// };
    /// use ndarray::s;
    /// use nshare::ToNdarray2;
    ///
    /// // Note: from_vec takes data column-by-column !
    /// let m = Matrix::<f32, Dynamic, Dynamic, _>::from_vec(
    ///     3,
    ///     4,
    ///     vec![0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 1.1, 1.2, 1.3, 1.5, 1.6, 1.7],
    /// );
    /// let arr = m.into_ndarray2();
    /// assert!(arr.slice(s![.., 0]).iter().eq(&[0.1, 0.2, 0.3]));
    /// assert!(arr.slice(s![0, ..]).iter().eq(&[0.1, 0.5, 1.1, 1.5]));
    /// ```
    impl<'a, N: Scalar> ToNdarray2 for Matrix<N, Dynamic, Dynamic, VecStorage<N, Dynamic, Dynamic>>
    where
        DefaultAllocator: Allocator<N, Dynamic, Dynamic, Buffer = VecStorage<N, Dynamic, Dynamic>>,
    {
        type Out = Array2<N>;

        fn into_ndarray2(self) -> Self::Out {
            Array2::from_shape_vec(self.shape().strides(self.strides()), self.data.into()).unwrap()
        }
    }
}
