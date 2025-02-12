//basically all from image crate but modified to add our own enum variants
//https://docs.rs/image/latest/src/image/dynimage.rs.html

use core::panic;
use std::{
    cmp::max,
    io::{self, Seek, Write},
    path::Path,
};

#[cfg(feature = "gif")]
use image::codecs::gif;
#[cfg(feature = "png")]
use image::codecs::png;

use crate::free_functions;
use image::{
    buffer::ConvertBuffer,
    error::{ImageError, ImageResult, LimitError, LimitErrorKind, ParameterError, ParameterErrorKind},
    flat::FlatSamples,
    imageops, ColorType, GenericImageView, GrayAlphaImage, GrayImage, ImageBuffer, ImageDecoder, ImageEncoder, ImageFormat, Luma, LumaA, Pixel,
    Primitive, Rgb, Rgb32FImage, RgbImage, Rgba, Rgba32FImage, RgbaImage,
};

pub type Gray16Image = ImageBuffer<Luma<u16>, Vec<u16>>;
pub type Gray32FImage = ImageBuffer<Luma<f32>, Vec<f32>>;
pub type GrayAlpha16Image = ImageBuffer<LumaA<u16>, Vec<u16>>;
pub type Rgb16Image = ImageBuffer<Rgb<u16>, Vec<u16>>;
pub type Rgba16Image = ImageBuffer<Rgba<u16>, Vec<u16>>;

/// Calculates the width and height an image should be resized to.
/// This preserves aspect ratio, and based on the `fill` parameter
/// will either fill the dimensions to fit inside the smaller constraint
/// (will overflow the specified bounds on one axis to preserve
/// aspect ratio), or will shrink so that both dimensions are
/// completely contained within the given `width` and `height`,
/// with empty space on one axis.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub(crate) fn resize_dimensions(width: u32, height: u32, nwidth: u32, nheight: u32, fill: bool) -> (u32, u32) {
    let wratio = f64::from(nwidth) / f64::from(width);
    let hratio = f64::from(nheight) / f64::from(height);

    let ratio = if fill { f64::max(wratio, hratio) } else { f64::min(wratio, hratio) };

    let nw = max((f64::from(width) * ratio).round() as u64, 1);
    let nh = max((f64::from(height) * ratio).round() as u64, 1);

    if nw > u64::from(u32::MAX) {
        let ratio = f64::from(u32::MAX) / f64::from(width);
        (u32::MAX, max((f64::from(height) * ratio).round() as u32, 1))
    } else if nh > u64::from(u32::MAX) {
        let ratio = f64::from(u32::MAX) / f64::from(height);
        (max((f64::from(width) * ratio).round() as u32, 1), u32::MAX)
    } else {
        (nw as u32, nh as u32)
    }
}

/// Reads all of the bytes of a decoder into a Vec<T>. No particular alignment
/// of the output buffer is guaranteed.
///
/// Panics if there isn't enough memory to decode the image.
pub(crate) fn decoder_to_vec<T>(decoder: impl ImageDecoder) -> ImageResult<Vec<T>>
where
    T: Primitive + bytemuck::Pod,
{
    let total_bytes = usize::try_from(decoder.total_bytes());
    if total_bytes.is_err() || total_bytes.unwrap() > isize::MAX as usize {
        return Err(ImageError::Limits(LimitError::from_kind(LimitErrorKind::InsufficientMemory)));
    }

    let mut buf = vec![num_traits::Zero::zero(); total_bytes.unwrap() / std::mem::size_of::<T>()];
    decoder.read_image(bytemuck::cast_slice_mut(buf.as_mut_slice()))?;
    Ok(buf)
}

// /// Provides color conversions for the different pixel types.
// pub trait FromColor<Other> {
//     /// Changes `self` to represent `Other` in the color space of `Self`
//     #[allow(clippy::wrong_self_convention)]
//     fn from_color(&mut self, _: &Other);
// }

// /// Copy-based conversions to target pixel types using `FromColor`.
// // FIXME: this trait should be removed and replaced with real color space
// models // rather than assuming sRGB.
// pub trait IntoColor<Other> {
//     /// Constructs a pixel of the target type and converts this pixel into
// it.     #[allow(clippy::wrong_self_convention)]
//     fn into_color(&self) -> Other;
// }

// impl<O, S> IntoColor<O> for S
// where
//     O: Pixel + FromColor<S>,
// {
//     #[allow(clippy::wrong_self_convention)]
//     fn into_color(&self) -> O {
//         // Note we cannot use Pixel::CHANNELS_COUNT here to directly
// construct         // the pixel due to a current bug/limitation of consts.
//         #[allow(deprecated)]
//         let mut pix = O::from_channels(Zero::zero(), Zero::zero(),
// Zero::zero(), Zero::zero());         pix.from_color(self);
//         pix
//     }
// }

/// A Dynamic Image
///
/// This represents a _matrix_ of _pixels_ which are _convertible_ from and to
/// an _RGBA_ representation. More variants that adhere to these principles may
/// get added in the future, in particular to cover other combinations typically
/// used.
///
/// # Usage
///
/// This type can act as a converter between specific `ImageBuffer` instances.
///
/// ```
/// use image::{DynamicImage, GrayImage, RgbImage};
///
/// let rgb: RgbImage = RgbImage::new(10, 10);
/// let luma: GrayImage = DynamicImage::ImageRgb8(rgb).into_luma8();
/// ```
///
/// # Design
///
/// There is no goal to provide an all-encompassing type with all possible
/// memory layouts. This would hardly be feasible as a simple enum, due to the
/// sheer number of combinations of channel kinds, channel order, and bit depth.
/// Rather, this type provides an opinionated selection with normalized channel
/// order which can store common pixel values without loss.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DynImage {
    /// Each pixel in this image is 8-bit Luma
    ImageLuma8(GrayImage),

    /// Each pixel in this image is 8-bit Luma with alpha
    ImageLumaA8(GrayAlphaImage),

    /// Each pixel in this image is 8-bit Rgb
    ImageRgb8(RgbImage),

    /// Each pixel in this image is 8-bit Rgb with alpha
    ImageRgba8(RgbaImage),

    /// Each pixel in this image is 16-bit Luma
    ImageLuma16(Gray16Image),

    /// Each pixel in this image is 16-bit Luma with alpha
    ImageLumaA16(GrayAlpha16Image),

    /// Each pixel in this image is 32-bit float Luma
    ImageLuma32F(Gray32FImage),

    /// Each pixel in this image is 16-bit Rgb
    ImageRgb16(Rgb16Image),

    /// Each pixel in this image is 16-bit Rgb with alpha
    ImageRgba16(Rgba16Image),

    /// Each pixel in this image is 32-bit float Rgb
    ImageRgb32F(Rgb32FImage),

    /// Each pixel in this image is 32-bit float Rgb with alpha
    ImageRgba32F(Rgba32FImage),
}

macro_rules! dynamic_map(
        ($dynimage: expr, $image: pat => $action: expr) => ({
            use DynImage::*;
            match $dynimage {
                ImageLuma8($image) => ImageLuma8($action),
                ImageLumaA8($image) => ImageLumaA8($action),
                ImageRgb8($image) => ImageRgb8($action),
                ImageRgba8($image) => ImageRgba8($action),
                ImageLuma16($image) => ImageLuma16($action),
                ImageLumaA16($image) => ImageLumaA16($action),
                ImageLuma32F($image) => ImageLuma32F($action),
                ImageRgb16($image) => ImageRgb16($action),
                ImageRgba16($image) => ImageRgba16($action),
                ImageRgb32F($image) => ImageRgb32F($action),
                ImageRgba32F($image) => ImageRgba32F($action),
            }
        });

        ($dynimage: expr, $image:pat_param, $action: expr) => (
            match $dynimage {
                DynImage::ImageLuma8($image) => $action,
                DynImage::ImageLumaA8($image) => $action,
                DynImage::ImageRgb8($image) => $action,
                DynImage::ImageRgba8($image) => $action,
                DynImage::ImageLuma16($image) => $action,
                DynImage::ImageLumaA16($image) => $action,
                DynImage::ImageLuma32F($image) => $action,
                DynImage::ImageRgb16($image) => $action,
                DynImage::ImageRgba16($image) => $action,
                DynImage::ImageRgb32F($image) => $action,
                DynImage::ImageRgba32F($image) => $action,
            }
        );
);

macro_rules! dynamic_map_img(
        ($dynimage: expr, $image: pat => $action: expr) => ({
            use image::DynamicImage::*;
            match $dynimage {
                ImageLuma8($image) => ImageLuma8($action),
                ImageLumaA8($image) => ImageLumaA8($action),
                ImageRgb8($image) => ImageRgb8($action),
                ImageRgba8($image) => ImageRgba8($action),
                ImageLuma16($image) => ImageLuma16($action),
                ImageLumaA16($image) => ImageLumaA16($action),
                ImageLuma32F($image) => ImageLuma32F($action),
                ImageRgb16($image) => ImageRgb16($action),
                ImageRgba16($image) => ImageRgba16($action),
                ImageRgb32F($image) => ImageRgb32F($action),
                ImageRgba32F($image) => ImageRgba32F($action),
            }
        });

        ($dynimage: expr, $image:pat_param, $action: expr) => (
            match $dynimage {
                image::DynamicImage::ImageLuma8($image) => $action,
                image::DynamicImage::ImageLumaA8($image) => $action,
                image::DynamicImage::ImageRgb8($image) => $action,
                image::DynamicImage::ImageRgba8($image) => $action,
                image::DynamicImage::ImageLuma16($image) => $action,
                image::DynamicImage::ImageLumaA16($image) => $action,
                // image::DynamicImage::ImageLuma32F($image) => $action,
                image::DynamicImage::ImageRgb16($image) => $action,
                image::DynamicImage::ImageRgba16($image) => $action,
                image::DynamicImage::ImageRgb32F($image) => $action,
                image::DynamicImage::ImageRgba32F($image) => $action,
                _ => panic!("Unkown format for dynamic_map")
            }
        );
);

impl DynImage {
    /// Creates a dynamic image backed by a buffer depending on
    /// the color type given.
    #[allow(clippy::enum_glob_use)]
    pub fn new(w: u32, h: u32, color: ColorType) -> DynImage {
        // use crate::color::ColorType::*;
        use image::ColorType::*;
        match color {
            L8 => Self::new_luma8(w, h),
            La8 => Self::new_luma_a8(w, h),
            Rgb8 => Self::new_rgb8(w, h),
            Rgba8 => Self::new_rgba8(w, h),
            L16 => Self::new_luma16(w, h),
            La16 => Self::new_luma_a16(w, h),
            // L32F => Self::new_luma32f(w, h),
            Rgb16 => Self::new_rgb16(w, h),
            Rgba16 => Self::new_rgba16(w, h),
            Rgb32F => Self::new_rgb32f(w, h),
            Rgba32F => Self::new_rgba32f(w, h),
            _ => panic!("Not supported image type in the image-rs crate"),
        }
    }

    /// Creates a dynamic image backed by a buffer of gray pixels.
    pub fn new_luma8(w: u32, h: u32) -> DynImage {
        DynImage::ImageLuma8(ImageBuffer::new(w, h))
    }

    /// Creates a dynamic image backed by a buffer of gray
    /// pixels with transparency.
    pub fn new_luma_a8(w: u32, h: u32) -> DynImage {
        DynImage::ImageLumaA8(ImageBuffer::new(w, h))
    }

    /// Creates a dynamic image backed by a buffer of RGB pixels.
    pub fn new_rgb8(w: u32, h: u32) -> DynImage {
        DynImage::ImageRgb8(ImageBuffer::new(w, h))
    }

    /// Creates a dynamic image backed by a buffer of RGBA pixels.
    pub fn new_rgba8(w: u32, h: u32) -> DynImage {
        DynImage::ImageRgba8(ImageBuffer::new(w, h))
    }

    /// Creates a dynamic image backed by a buffer of gray pixels.
    pub fn new_luma16(w: u32, h: u32) -> DynImage {
        DynImage::ImageLuma16(ImageBuffer::new(w, h))
    }

    /// Creates a dynamic image backed by a buffer of gray
    /// pixels with transparency.
    pub fn new_luma_a16(w: u32, h: u32) -> DynImage {
        DynImage::ImageLumaA16(ImageBuffer::new(w, h))
    }

    /// Creates a dynamic image backed by a buffer of gray pixels.
    pub fn new_luma32f(w: u32, h: u32) -> DynImage {
        DynImage::ImageLuma32F(ImageBuffer::new(w, h))
    }

    /// Creates a dynamic image backed by a buffer of RGB pixels.
    pub fn new_rgb16(w: u32, h: u32) -> DynImage {
        DynImage::ImageRgb16(ImageBuffer::new(w, h))
    }

    /// Creates a dynamic image backed by a buffer of RGBA pixels.
    pub fn new_rgba16(w: u32, h: u32) -> DynImage {
        DynImage::ImageRgba16(ImageBuffer::new(w, h))
    }

    /// Creates a dynamic image backed by a buffer of RGB pixels.
    pub fn new_rgb32f(w: u32, h: u32) -> DynImage {
        DynImage::ImageRgb32F(ImageBuffer::new(w, h))
    }

    /// Creates a dynamic image backed by a buffer of RGBA pixels.
    pub fn new_rgba32f(w: u32, h: u32) -> DynImage {
        DynImage::ImageRgba32F(ImageBuffer::new(w, h))
    }

    /// Decodes an encoded image into a dynamic image.
    #[allow(clippy::missing_errors_doc)]
    pub fn from_decoder(decoder: impl ImageDecoder) -> ImageResult<Self> {
        decoder_to_image(decoder)
    }

    /// Returns a copy of this image as an RGB image.
    pub fn to_rgb8(&self) -> RgbImage {
        dynamic_map!(*self, ref p, p.convert())
    }

    /// Returns a copy of this image as an RGB image.
    pub fn to_rgb16(&self) -> Rgb16Image {
        dynamic_map!(*self, ref p, p.convert())
    }

    /// Returns a copy of this image as an RGB image.
    pub fn to_rgb32f(&self) -> Rgb32FImage {
        dynamic_map!(*self, ref p, p.convert())
    }

    /// Returns a copy of this image as an RGBA image.
    pub fn to_rgba8(&self) -> RgbaImage {
        dynamic_map!(*self, ref p, p.convert())
    }

    /// Returns a copy of this image as an RGBA image.
    pub fn to_rgba16(&self) -> Rgba16Image {
        dynamic_map!(*self, ref p, p.convert())
    }

    /// Returns a copy of this image as an RGBA image.
    pub fn to_rgba32f(&self) -> Rgba32FImage {
        dynamic_map!(*self, ref p, p.convert())
    }

    /// Returns a copy of this image as a Luma image.
    pub fn to_luma8(&self) -> GrayImage {
        dynamic_map!(*self, ref p, p.convert())
    }

    /// Returns a copy of this image as a Luma image.
    pub fn to_luma16(&self) -> Gray16Image {
        dynamic_map!(*self, ref p, p.convert())
    }

    /// Returns a copy of this image as a Luma image.
    pub fn to_luma32f(&self) -> ImageBuffer<Luma<f32>, Vec<f32>> {
        dynamic_map!(*self, ref p, p.convert())
    }

    /// Returns a copy of this image as a `LumaA` image.
    pub fn to_luma_alpha8(&self) -> GrayAlphaImage {
        dynamic_map!(*self, ref p, p.convert())
    }

    /// Returns a copy of this image as a `LumaA` image.
    pub fn to_luma_alpha16(&self) -> GrayAlpha16Image {
        dynamic_map!(*self, ref p, p.convert())
    }

    /// Returns a copy of this image as a `LumaA` image.
    pub fn to_luma_alpha32f(&self) -> ImageBuffer<LumaA<f32>, Vec<f32>> {
        dynamic_map!(*self, ref p, p.convert())
    }

    /// Consume the image and returns a RGB image.
    ///
    /// If the image was already the correct format, it is returned as is.
    /// Otherwise, a copy is created.
    pub fn into_rgb8(self) -> RgbImage {
        match self {
            DynImage::ImageRgb8(x) => x,
            x => x.to_rgb8(),
        }
    }

    /// Consume the image and returns a RGB image.
    ///
    /// If the image was already the correct format, it is returned as is.
    /// Otherwise, a copy is created.
    pub fn into_rgb16(self) -> Rgb16Image {
        match self {
            DynImage::ImageRgb16(x) => x,
            x => x.to_rgb16(),
        }
    }

    /// Consume the image and returns a RGB image.
    ///
    /// If the image was already the correct format, it is returned as is.
    /// Otherwise, a copy is created.
    pub fn into_rgb32f(self) -> Rgb32FImage {
        match self {
            DynImage::ImageRgb32F(x) => x,
            x => x.to_rgb32f(),
        }
    }

    /// Consume the image and returns a RGBA image.
    ///
    /// If the image was already the correct format, it is returned as is.
    /// Otherwise, a copy is created.
    pub fn into_rgba8(self) -> RgbaImage {
        match self {
            DynImage::ImageRgba8(x) => x,
            x => x.to_rgba8(),
        }
    }

    /// Consume the image and returns a RGBA image.
    ///
    /// If the image was already the correct format, it is returned as is.
    /// Otherwise, a copy is created.
    pub fn into_rgba16(self) -> Rgba16Image {
        match self {
            DynImage::ImageRgba16(x) => x,
            x => x.to_rgba16(),
        }
    }

    /// Consume the image and returns a RGBA image.
    ///
    /// If the image was already the correct format, it is returned as is.
    /// Otherwise, a copy is created.
    pub fn into_rgba32f(self) -> Rgba32FImage {
        match self {
            DynImage::ImageRgba32F(x) => x,
            x => x.to_rgba32f(),
        }
    }

    /// Consume the image and returns a Luma image.
    ///
    /// If the image was already the correct format, it is returned as is.
    /// Otherwise, a copy is created.
    pub fn into_luma8(self) -> GrayImage {
        match self {
            DynImage::ImageLuma8(x) => x,
            x => x.to_luma8(),
        }
    }

    /// Consume the image and returns a Luma image.
    ///
    /// If the image was already the correct format, it is returned as is.
    /// Otherwise, a copy is created.
    pub fn into_luma16(self) -> Gray16Image {
        match self {
            DynImage::ImageLuma16(x) => x,
            x => x.to_luma16(),
        }
    }

    /// Consume the image and returns a `LumaA` image.
    ///
    /// If the image was already the correct format, it is returned as is.
    /// Otherwise, a copy is created.
    pub fn into_luma_alpha8(self) -> GrayAlphaImage {
        match self {
            DynImage::ImageLumaA8(x) => x,
            x => x.to_luma_alpha8(),
        }
    }

    /// Consume the image and returns a `LumaA` image.
    ///
    /// If the image was already the correct format, it is returned as is.
    /// Otherwise, a copy is created.
    pub fn into_luma_alpha16(self) -> GrayAlpha16Image {
        match self {
            DynImage::ImageLumaA16(x) => x,
            x => x.to_luma_alpha16(),
        }
    }

    /// Return a cut-out of this image delimited by the bounding rectangle.
    ///
    /// Note: this method does *not* modify the object,
    /// and its signature will be replaced with `crop_imm()`'s in the 0.24
    /// release
    #[must_use]
    pub fn crop(&mut self, x: u32, y: u32, width: u32, height: u32) -> DynImage {
        dynamic_map!(*self, ref mut p => imageops::crop(p, x, y, width, height).to_image())
    }

    /// Return a cut-out of this image delimited by the bounding rectangle.
    #[must_use]
    pub fn crop_imm(&self, x: u32, y: u32, width: u32, height: u32) -> DynImage {
        dynamic_map!(*self, ref p => imageops::crop_imm(p, x, y, width, height).to_image())
    }

    /// Return a reference to an 8bit RGB image
    pub fn as_rgb8(&self) -> Option<&RgbImage> {
        match *self {
            DynImage::ImageRgb8(ref p) => Some(p),
            _ => None,
        }
    }

    /// Return a mutable reference to an 8bit RGB image
    pub fn as_mut_rgb8(&mut self) -> Option<&mut RgbImage> {
        match *self {
            DynImage::ImageRgb8(ref mut p) => Some(p),
            _ => None,
        }
    }

    /// Return a reference to an 8bit RGBA image
    pub fn as_rgba8(&self) -> Option<&RgbaImage> {
        match *self {
            DynImage::ImageRgba8(ref p) => Some(p),
            _ => None,
        }
    }

    /// Return a mutable reference to an 8bit RGBA image
    pub fn as_mut_rgba8(&mut self) -> Option<&mut RgbaImage> {
        match *self {
            DynImage::ImageRgba8(ref mut p) => Some(p),
            _ => None,
        }
    }

    /// Return a reference to an 8bit Grayscale image
    pub fn as_luma8(&self) -> Option<&GrayImage> {
        match *self {
            DynImage::ImageLuma8(ref p) => Some(p),
            _ => None,
        }
    }

    /// Return a mutable reference to an 8bit Grayscale image
    pub fn as_mut_luma8(&mut self) -> Option<&mut GrayImage> {
        match *self {
            DynImage::ImageLuma8(ref mut p) => Some(p),
            _ => None,
        }
    }

    /// Return a reference to an 8bit Grayscale image with an alpha channel
    pub fn as_luma_alpha8(&self) -> Option<&GrayAlphaImage> {
        match *self {
            DynImage::ImageLumaA8(ref p) => Some(p),
            _ => None,
        }
    }

    /// Return a mutable reference to an 8bit Grayscale image with an alpha
    /// channel
    pub fn as_mut_luma_alpha8(&mut self) -> Option<&mut GrayAlphaImage> {
        match *self {
            DynImage::ImageLumaA8(ref mut p) => Some(p),
            _ => None,
        }
    }

    /// Return a reference to an 16bit RGB image
    pub fn as_rgb16(&self) -> Option<&Rgb16Image> {
        match *self {
            DynImage::ImageRgb16(ref p) => Some(p),
            _ => None,
        }
    }

    /// Return a mutable reference to an 16bit RGB image
    pub fn as_mut_rgb16(&mut self) -> Option<&mut Rgb16Image> {
        match *self {
            DynImage::ImageRgb16(ref mut p) => Some(p),
            _ => None,
        }
    }

    /// Return a reference to an 16bit RGBA image
    pub fn as_rgba16(&self) -> Option<&Rgba16Image> {
        match *self {
            DynImage::ImageRgba16(ref p) => Some(p),
            _ => None,
        }
    }

    /// Return a mutable reference to an 16bit RGBA image
    pub fn as_mut_rgba16(&mut self) -> Option<&mut Rgba16Image> {
        match *self {
            DynImage::ImageRgba16(ref mut p) => Some(p),
            _ => None,
        }
    }

    /// Return a reference to an 32bit RGB image
    pub fn as_rgb32f(&self) -> Option<&Rgb32FImage> {
        match *self {
            DynImage::ImageRgb32F(ref p) => Some(p),
            _ => None,
        }
    }

    /// Return a mutable reference to an 32bit RGB image
    pub fn as_mut_rgb32f(&mut self) -> Option<&mut Rgb32FImage> {
        match *self {
            DynImage::ImageRgb32F(ref mut p) => Some(p),
            _ => None,
        }
    }

    /// Return a reference to an 32bit RGBA image
    pub fn as_rgba32f(&self) -> Option<&Rgba32FImage> {
        match *self {
            DynImage::ImageRgba32F(ref p) => Some(p),
            _ => None,
        }
    }

    /// Return a mutable reference to an 16bit RGBA image
    pub fn as_mut_rgba32f(&mut self) -> Option<&mut Rgba32FImage> {
        match *self {
            DynImage::ImageRgba32F(ref mut p) => Some(p),
            _ => None,
        }
    }

    /// Return a reference to an 16bit Grayscale image
    pub fn as_luma16(&self) -> Option<&Gray16Image> {
        match *self {
            DynImage::ImageLuma16(ref p) => Some(p),
            _ => None,
        }
    }

    /// Return a mutable reference to an 16bit Grayscale image
    pub fn as_mut_luma16(&mut self) -> Option<&mut Gray16Image> {
        match *self {
            DynImage::ImageLuma16(ref mut p) => Some(p),
            _ => None,
        }
    }

    /// Return a reference to an 16bit Grayscale image with an alpha channel
    pub fn as_luma_alpha16(&self) -> Option<&GrayAlpha16Image> {
        match *self {
            DynImage::ImageLumaA16(ref p) => Some(p),
            _ => None,
        }
    }

    /// Return a mutable reference to an 16bit Grayscale image with an alpha
    /// channel
    pub fn as_mut_luma_alpha16(&mut self) -> Option<&mut GrayAlpha16Image> {
        match *self {
            DynImage::ImageLumaA16(ref mut p) => Some(p),
            _ => None,
        }
    }

    /// Return a view on the raw sample buffer for 8 bit per channel images.
    pub fn as_flat_samples_u8(&self) -> Option<FlatSamples<&[u8]>> {
        match *self {
            DynImage::ImageLuma8(ref p) => Some(p.as_flat_samples()),
            DynImage::ImageLumaA8(ref p) => Some(p.as_flat_samples()),
            DynImage::ImageRgb8(ref p) => Some(p.as_flat_samples()),
            DynImage::ImageRgba8(ref p) => Some(p.as_flat_samples()),
            _ => None,
        }
    }

    /// Return a view on the raw sample buffer for 16 bit per channel images.
    pub fn as_flat_samples_u16(&self) -> Option<FlatSamples<&[u16]>> {
        match *self {
            DynImage::ImageLuma16(ref p) => Some(p.as_flat_samples()),
            DynImage::ImageLumaA16(ref p) => Some(p.as_flat_samples()),
            DynImage::ImageRgb16(ref p) => Some(p.as_flat_samples()),
            DynImage::ImageRgba16(ref p) => Some(p.as_flat_samples()),
            _ => None,
        }
    }

    /// Return a view on the raw sample buffer for 32bit per channel images.
    pub fn as_flat_samples_f32(&self) -> Option<FlatSamples<&[f32]>> {
        match *self {
            DynImage::ImageLuma32F(ref p) => Some(p.as_flat_samples()),
            DynImage::ImageRgb32F(ref p) => Some(p.as_flat_samples()),
            DynImage::ImageRgba32F(ref p) => Some(p.as_flat_samples()),
            _ => None,
        }
    }

    /// Return this image's pixels as a native endian byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        // we can do this because every variant contains an `ImageBuffer<_, Vec<_>>`
        dynamic_map!(*self, ref image_buffer, bytemuck::cast_slice(image_buffer.as_raw().as_ref()))
    }

    // TODO: choose a name under which to expose?
    fn inner_bytes(&self) -> &[u8] {
        // we can do this because every variant contains an `ImageBuffer<_, Vec<_>>`
        // let len = Self::image_buffer_len(self.width, self.height).unwrap();
        // let channels =<P as Pixel>::CHANNEL_COUNT
        // let channel_count = self.color().channel_count() as u32;
        // let len = channel_count * self.width() * self.height();
        // let len = len as usize;

        // let data = dynamic_map!(
        //     *self,
        //     ref image_buffer,
        //     bytemuck::cast_slice(image_buffer.as_raw())
        // );

        //TODO check that this is correct since the previous implementation also sliced
        // with the length of the data
        dynamic_map!(*self, ref image_buffer, bytemuck::cast_slice(image_buffer.as_raw()))

        // dynamic_map!(
        //     *self,
        //     ref image_buffer,
        //     {
        //         let samples = image_buffer.as_flat_samples();
        //         let img_slice = samples.image_slice().unwrap();
        //         bytemuck::cast_slice(img_slice)
        //     } // bytemuck::cast_slice(image_buffer.as_flat_samples().
        // image_slice().unwrap()) )
    }

    /// Return this image's pixels as a byte vector. If the `ImageBuffer`
    /// container is `Vec<u8>`, this operation is free. Otherwise, a copy
    /// is returned.
    pub fn into_bytes(self) -> Vec<u8> {
        // we can do this because every variant contains an `ImageBuffer<_, Vec<_>>`
        dynamic_map!(self, image_buffer, {
            match bytemuck::allocation::try_cast_vec(image_buffer.into_raw()) {
                Ok(vec) => vec,
                Err((_, vec)) => {
                    // Fallback: vector requires an exact alignment and size match
                    // Reuse of the allocation as done in the Ok branch only works if the
                    // underlying container is exactly Vec<u8> (or compatible but that's the only
                    // alternative at the time of writing).
                    // In all other cases we must allocate a new vector with the 'same' contents.
                    bytemuck::cast_slice(&vec).to_owned()
                }
            }
        })
    }

    /// Return this image's color type.
    pub fn color(&self) -> ColorType {
        match *self {
            DynImage::ImageLuma8(_) => ColorType::L8,
            DynImage::ImageLumaA8(_) => ColorType::La8,
            DynImage::ImageRgb8(_) => ColorType::Rgb8,
            DynImage::ImageRgba8(_) => ColorType::Rgba8,
            DynImage::ImageLuma16(_) => ColorType::L16,
            DynImage::ImageLumaA16(_) => ColorType::La16,
            DynImage::ImageLuma32F(_) => {
                panic!("Not a supported ColorType in the image-rs crate. If you need the number of channels use dynimage,channels()")
            } //Would need to extend colortype which is a bit of a pain
            DynImage::ImageRgb16(_) => ColorType::Rgb16,
            DynImage::ImageRgba16(_) => ColorType::Rgba16,
            DynImage::ImageRgb32F(_) => ColorType::Rgb32F,
            DynImage::ImageRgba32F(_) => ColorType::Rgba32F,
        }
    }

    /// Returns the width of the underlying image
    pub fn width(&self) -> u32 {
        dynamic_map!(*self, ref p, { p.width() })
    }

    /// Returns the height of the underlying image
    pub fn height(&self) -> u32 {
        dynamic_map!(*self, ref p, { p.height() })
    }

    pub fn channels(&self) -> u32 {
        match *self {
            DynImage::ImageLuma8(_) | DynImage::ImageLuma16(_) | DynImage::ImageLuma32F(_) => 1,
            DynImage::ImageLumaA8(_) | DynImage::ImageLumaA16(_) => 2,
            DynImage::ImageRgb8(_) | DynImage::ImageRgb16(_) | DynImage::ImageRgb32F(_) => 3,
            DynImage::ImageRgba8(_) | DynImage::ImageRgba16(_) | DynImage::ImageRgba32F(_) => 4,
        }
    }

    /// Return a grayscale version of this image.
    /// Returns `Luma` images in most cases. However, for `f32` images,
    /// this will return a grayscale `Rgb/Rgba` image instead.
    #[must_use]
    pub fn grayscale(&self) -> DynImage {
        match *self {
            DynImage::ImageLuma8(ref p) => DynImage::ImageLuma8(p.clone()),
            DynImage::ImageLumaA8(ref p) => DynImage::ImageLumaA8(imageops::grayscale_alpha(p)),
            DynImage::ImageRgb8(ref p) => DynImage::ImageLuma8(imageops::grayscale(p)),
            DynImage::ImageRgba8(ref p) => DynImage::ImageLumaA8(imageops::grayscale_alpha(p)),
            DynImage::ImageLuma16(ref p) => DynImage::ImageLuma16(p.clone()),
            DynImage::ImageLumaA16(ref p) => DynImage::ImageLumaA16(imageops::grayscale_alpha(p)),
            DynImage::ImageLuma32F(ref p) => DynImage::ImageLuma32F(p.clone()),
            DynImage::ImageRgb16(ref p) => DynImage::ImageLuma16(imageops::grayscale(p)),
            DynImage::ImageRgba16(ref p) => DynImage::ImageLumaA16(imageops::grayscale_alpha(p)),
            DynImage::ImageRgb32F(ref p) => DynImage::ImageRgb32F(imageops::grayscale_with_type(p)),
            DynImage::ImageRgba32F(ref p) => DynImage::ImageRgba32F(imageops::grayscale_with_type_alpha(p)),
        }
    }

    /// Invert the colors of this image.
    /// This method operates inplace.
    pub fn invert(&mut self) {
        dynamic_map!(*self, ref mut p, imageops::invert(p));
    }

    /// Resize this image using the specified filter algorithm.
    /// Returns a new image. The image's aspect ratio is preserved.
    /// The image is scaled to the maximum possible size that fits
    /// within the bounds specified by `nwidth` and `nheight`.
    #[must_use]
    pub fn resize(&self, nwidth: u32, nheight: u32, filter: imageops::FilterType) -> DynImage {
        if (nwidth, nheight) == self.dimensions() {
            return self.clone();
        }
        let (width2, height2) = resize_dimensions(self.width(), self.height(), nwidth, nheight, false);

        self.resize_exact(width2, height2, filter)
    }

    /// Resize this image using the specified filter algorithm.
    /// Returns a new image. Does not preserve aspect ratio.
    /// `nwidth` and `nheight` are the new image's dimensions
    #[must_use]
    pub fn resize_exact(&self, nwidth: u32, nheight: u32, filter: imageops::FilterType) -> DynImage {
        dynamic_map!(*self, ref p => imageops::resize(p, nwidth, nheight, filter))
    }

    /// Scale this image down to fit within a specific size.
    /// Returns a new image. The image's aspect ratio is preserved.
    /// The image is scaled to the maximum possible size that fits
    /// within the bounds specified by `nwidth` and `nheight`.
    ///
    /// This method uses a fast integer algorithm where each source
    /// pixel contributes to exactly one target pixel.
    /// May give aliasing artifacts if new size is close to old size.
    #[must_use]
    pub fn thumbnail(&self, nwidth: u32, nheight: u32) -> DynImage {
        let (width2, height2) = resize_dimensions(self.width(), self.height(), nwidth, nheight, false);
        self.thumbnail_exact(width2, height2)
    }

    /// Scale this image down to a specific size.
    /// Returns a new image. Does not preserve aspect ratio.
    /// `nwidth` and `nheight` are the new image's dimensions.
    /// This method uses a fast integer algorithm where each source
    /// pixel contributes to exactly one target pixel.
    /// May give aliasing artifacts if new size is close to old size.
    #[must_use]
    pub fn thumbnail_exact(&self, nwidth: u32, nheight: u32) -> DynImage {
        dynamic_map!(*self, ref p => imageops::thumbnail(p, nwidth, nheight))
    }

    /// Resize this image using the specified filter algorithm.
    /// Returns a new image. The image's aspect ratio is preserved.
    /// The image is scaled to the maximum possible size that fits
    /// within the larger (relative to aspect ratio) of the bounds
    /// specified by `nwidth` and `nheight`, then cropped to
    /// fit within the other bound.
    #[must_use]
    pub fn resize_to_fill(&self, nwidth: u32, nheight: u32, filter: imageops::FilterType) -> DynImage {
        let (width2, height2) = resize_dimensions(self.width(), self.height(), nwidth, nheight, true);

        let mut intermediate = self.resize_exact(width2, height2, filter);
        let (iwidth, iheight) = intermediate.dimensions();
        let ratio = u64::from(iwidth) * u64::from(nheight);
        let nratio = u64::from(nwidth) * u64::from(iheight);

        if nratio > ratio {
            intermediate.crop(0, (iheight - nheight) / 2, nwidth, nheight)
        } else {
            intermediate.crop((iwidth - nwidth) / 2, 0, nwidth, nheight)
        }
    }

    /// Performs a Gaussian blur on this image.
    /// `sigma` is a measure of how much to blur by.
    #[must_use]
    pub fn blur(&self, sigma: f32) -> DynImage {
        dynamic_map!(*self, ref p => imageops::blur(p, sigma))
    }

    /// Performs an unsharpen mask on this image.
    /// `sigma` is the amount to blur the image by.
    /// `threshold` is a control of how much to sharpen.
    ///
    /// See <https://en.wikipedia.org/wiki/Unsharp_masking#Digital_unsharp_masking>
    #[must_use]
    pub fn unsharpen(&self, sigma: f32, threshold: i32) -> DynImage {
        dynamic_map!(*self, ref p => imageops::unsharpen(p, sigma, threshold))
    }

    /// Filters this image with the specified 3x3 kernel.
    #[must_use]
    pub fn filter3x3(&self, kernel: &[f32]) -> DynImage {
        assert!(kernel.len() == 9, "filter must be 3 x 3");

        dynamic_map!(*self, ref p => imageops::filter3x3(p, kernel))
    }

    /// Adjust the contrast of this image.
    /// `contrast` is the amount to adjust the contrast by.
    /// Negative values decrease the contrast and positive values increase the
    /// contrast.
    #[must_use]
    pub fn adjust_contrast(&self, c: f32) -> DynImage {
        dynamic_map!(*self, ref p => imageops::contrast(p, c))
    }

    /// Brighten the pixels of this image.
    /// `value` is the amount to brighten each pixel by.
    /// Negative values decrease the brightness and positive values increase it.
    #[must_use]
    pub fn brighten(&self, value: i32) -> DynImage {
        dynamic_map!(*self, ref p => imageops::brighten(p, value))
    }

    /// Hue rotate the supplied image.
    /// `value` is the degrees to rotate each pixel by.
    /// 0 and 360 do nothing, the rest rotates by the given degree value.
    /// just like the css webkit filter hue-rotate(180)
    #[must_use]
    pub fn huerotate(&self, value: i32) -> DynImage {
        dynamic_map!(*self, ref p => imageops::huerotate(p, value))
    }

    /// Flip this image vertically
    #[must_use]
    pub fn flipv(&self) -> DynImage {
        dynamic_map!(*self, ref p => imageops::flip_vertical(p))
    }

    /// Flip this image horizontally
    #[must_use]
    pub fn fliph(&self) -> DynImage {
        dynamic_map!(*self, ref p => imageops::flip_horizontal(p))
    }

    /// Rotate this image 90 degrees clockwise.
    #[must_use]
    pub fn rotate90(&self) -> DynImage {
        dynamic_map!(*self, ref p => imageops::rotate90(p))
    }

    /// Rotate this image 180 degrees clockwise.
    #[must_use]
    pub fn rotate180(&self) -> DynImage {
        dynamic_map!(*self, ref p => imageops::rotate180(p))
    }

    /// Rotate this image 270 degrees clockwise.
    #[must_use]
    pub fn rotate270(&self) -> DynImage {
        dynamic_map!(*self, ref p => imageops::rotate270(p))
    }

    /// Encode this image and write it to ```w```.
    ///
    /// Assumes the writer is buffered. In most cases,
    /// you should wrap your writer in a `BufWriter` for best performance.
    #[allow(clippy::missing_errors_doc)]
    pub fn write_to<W: Write + Seek>(&self, w: &mut W, format: ImageFormat) -> ImageResult<()> {
        let bytes = self.inner_bytes();
        let (width, height) = self.dimensions();
        let color = self.color();

        // TODO do not repeat this match statement across the crate

        #[allow(deprecated)]
        match format {
            #[cfg(feature = "png")]
            ImageFormat::Png => {
                let p = png::PngEncoder::new(w);
                p.write_image(bytes, width, height, self.color().into())?;
                Ok(())
            }

            #[cfg(feature = "gif")]
            ImageFormat::Gif => {
                let mut g = gif::GifEncoder::new(w);
                g.encode_frame(image::Frame::new(self.to_rgba8()))?;
                Ok(())
            }

            format => write_buffer_with_format(w, bytes, width, height, color, format),
        }
    }

    /// Encode this image with the provided encoder.
    #[allow(clippy::missing_errors_doc)]
    pub fn write_with_encoder(&self, encoder: impl ImageEncoder) -> ImageResult<()> {
        // dynamic_map!(self, ref p, p.write_with_encoder(encoder))
        let img_dyn: image::DynamicImage = self.clone().try_into().unwrap();
        dynamic_map_img!(img_dyn, ref p, p.write_with_encoder(encoder))
    }

    /// Saves the buffer to a file at the path specified.
    ///
    /// The image format is derived from the file extension.
    #[allow(clippy::missing_errors_doc)]
    pub fn save<Q>(&self, path: Q) -> ImageResult<()>
    where
        Q: AsRef<Path>,
    {
        // dynamic_map!(*self, ref p, p.save(path))
        let img_dyn: image::DynamicImage = self.clone().try_into().unwrap();
        dynamic_map_img!(img_dyn, ref p, p.save(path))
    }

    /// Saves the buffer to a file at the specified path in
    /// the specified format.
    ///
    /// See [`save_buffer_with_format`](fn.save_buffer_with_format.html) for
    /// supported types.
    #[allow(clippy::missing_errors_doc)]
    pub fn save_with_format<Q>(&self, path: Q, format: ImageFormat) -> ImageResult<()>
    where
        Q: AsRef<Path>,
    {
        // dynamic_map!(*self, ref p, p.save_with_format(path, format))
        let img_dyn: image::DynamicImage = self.clone().try_into().unwrap();
        dynamic_map_img!(img_dyn, ref p, p.save_with_format(path, format))
    }
}

impl From<GrayImage> for DynImage {
    fn from(image: GrayImage) -> Self {
        DynImage::ImageLuma8(image)
    }
}

impl From<GrayAlphaImage> for DynImage {
    fn from(image: GrayAlphaImage) -> Self {
        DynImage::ImageLumaA8(image)
    }
}

impl From<RgbImage> for DynImage {
    fn from(image: RgbImage) -> Self {
        DynImage::ImageRgb8(image)
    }
}

impl From<RgbaImage> for DynImage {
    fn from(image: RgbaImage) -> Self {
        DynImage::ImageRgba8(image)
    }
}

impl From<Gray16Image> for DynImage {
    fn from(image: Gray16Image) -> Self {
        DynImage::ImageLuma16(image)
    }
}

impl From<GrayAlpha16Image> for DynImage {
    fn from(image: GrayAlpha16Image) -> Self {
        DynImage::ImageLumaA16(image)
    }
}

impl From<Rgb16Image> for DynImage {
    fn from(image: Rgb16Image) -> Self {
        DynImage::ImageRgb16(image)
    }
}

impl From<Rgba16Image> for DynImage {
    fn from(image: Rgba16Image) -> Self {
        DynImage::ImageRgba16(image)
    }
}

impl From<Rgb32FImage> for DynImage {
    fn from(image: Rgb32FImage) -> Self {
        DynImage::ImageRgb32F(image)
    }
}

impl From<Rgba32FImage> for DynImage {
    fn from(image: Rgba32FImage) -> Self {
        DynImage::ImageRgba32F(image)
    }
}

impl From<ImageBuffer<Luma<f32>, Vec<f32>>> for DynImage {
    fn from(image: ImageBuffer<Luma<f32>, Vec<f32>>) -> Self {
        DynImage::ImageRgb32F(image.convert())
    }
}

impl From<ImageBuffer<LumaA<f32>, Vec<f32>>> for DynImage {
    fn from(image: ImageBuffer<LumaA<f32>, Vec<f32>>) -> Self {
        DynImage::ImageRgba32F(image.convert())
    }
}

//convert to an from the image crate DynamicImage
impl TryFrom<image::DynamicImage> for DynImage {
    type Error = String;
    fn try_from(image: image::DynamicImage) -> Result<Self, Self::Error> {
        match image {
            image::DynamicImage::ImageLuma8(v) => Ok(DynImage::ImageLuma8(v)),
            image::DynamicImage::ImageLumaA8(v) => Ok(DynImage::ImageLumaA8(v)),
            image::DynamicImage::ImageRgb8(v) => Ok(DynImage::ImageRgb8(v)),
            image::DynamicImage::ImageRgba8(v) => Ok(DynImage::ImageRgba8(v)),
            image::DynamicImage::ImageLuma16(v) => Ok(DynImage::ImageLuma16(v)),
            image::DynamicImage::ImageLumaA16(v) => Ok(DynImage::ImageLumaA16(v)),
            image::DynamicImage::ImageRgb16(v) => Ok(DynImage::ImageRgb16(v)),
            image::DynamicImage::ImageRgba16(v) => Ok(DynImage::ImageRgba16(v)),
            image::DynamicImage::ImageRgb32F(v) => Ok(DynImage::ImageRgb32F(v)),
            image::DynamicImage::ImageRgba32F(v) => Ok(DynImage::ImageRgba32F(v)),
            _ => Err("Unkown format".to_string()),
        }
    }
}
// impl TryInto<DynamicImage> for image::DynamicImage {
//     type Error = String;

//     fn try_into(self) -> Result<DynamicImage, Self::Error> {
//         todo!()
//     }
// }
impl TryInto<image::DynamicImage> for DynImage {
    type Error = String;
    fn try_into(self) -> Result<image::DynamicImage, Self::Error> {
        match self {
            DynImage::ImageLuma8(v) => Ok(image::DynamicImage::ImageLuma8(v)),
            DynImage::ImageLumaA8(v) => Ok(image::DynamicImage::ImageLumaA8(v)),
            DynImage::ImageRgb8(v) => Ok(image::DynamicImage::ImageRgb8(v)),
            DynImage::ImageRgba8(v) => Ok(image::DynamicImage::ImageRgba8(v)),
            DynImage::ImageLuma16(v) => Ok(image::DynamicImage::ImageLuma16(v)),
            DynImage::ImageLumaA16(v) => Ok(image::DynamicImage::ImageLumaA16(v)),
            DynImage::ImageRgb16(v) => Ok(image::DynamicImage::ImageRgb16(v)),
            DynImage::ImageRgba16(v) => Ok(image::DynamicImage::ImageRgba16(v)),
            DynImage::ImageRgb32F(v) => Ok(image::DynamicImage::ImageRgb32F(v)),
            DynImage::ImageRgba32F(v) => Ok(image::DynamicImage::ImageRgba32F(v)),
            _ => panic!("Unknown format"),
        }
    }
}

#[allow(deprecated)]
impl GenericImageView for DynImage {
    type Pixel = image::Rgba<f32>; // TODO use f32 as default for best precision and unbounded color?

    fn dimensions(&self) -> (u32, u32) {
        dynamic_map!(*self, ref p, p.dimensions())
    }

    // fn bounds(&self) -> (u32, u32, u32, u32) {
    //     dynamic_map!(*self, ref p, p.bounds())
    // }

    #[allow(clippy::useless_conversion)]
    fn get_pixel(&self, x: u32, y: u32) -> Rgba<f32> {
        dynamic_map!(*self, ref p, {
            let pixel = p.get_pixel(x, y).to_rgba();
            Rgba::<f32>([f32::from(pixel.0[0]), f32::from(pixel.0[1]), f32::from(pixel.0[2]), f32::from(pixel.0[3])])
        })
        // unimplemented!() //because into_color is private
    }
}

// #[allow(deprecated)]
// impl GenericImage for DynamicImage {
//     fn put_pixel(&mut self, x: u32, y: u32, pixel: Rgba<u8>) {
//         match *self {
//             DynamicImage::ImageLuma8(ref mut p) => p.put_pixel(x, y,
// pixel.to_luma()),             DynamicImage::ImageLumaA8(ref mut p) =>
// p.put_pixel(x, y, pixel.to_luma_alpha()),
// DynamicImage::ImageRgb8(ref mut p) => p.put_pixel(x, y, pixel.to_rgb()),
//             DynamicImage::ImageRgba8(ref mut p) => p.put_pixel(x, y, pixel),
//             DynamicImage::ImageLuma16(ref mut p) => p.put_pixel(x, y,
// pixel.to_luma().into_color()),             DynamicImage::ImageLumaA16(ref mut
// p) => {                 p.put_pixel(x, y, pixel.to_luma_alpha().into_color())
//             }
//             DynamicImage::ImageRgb16(ref mut p) => p.put_pixel(x, y,
// pixel.to_rgb().into_color()),             DynamicImage::ImageRgba16(ref mut
// p) => p.put_pixel(x, y, pixel.into_color()),
// DynamicImage::ImageRgb32F(ref mut p) => p.put_pixel(x, y,
// pixel.to_rgb().into_color()),             DynamicImage::ImageRgba32F(ref mut
// p) => p.put_pixel(x, y, pixel.into_color()),         }
//     }

//     fn blend_pixel(&mut self, x: u32, y: u32, pixel: color::Rgba<u8>) {
//         match *self {
//             DynamicImage::ImageLuma8(ref mut p) => p.blend_pixel(x, y,
// pixel.to_luma()),             DynamicImage::ImageLumaA8(ref mut p) =>
// p.blend_pixel(x, y, pixel.to_luma_alpha()),
// DynamicImage::ImageRgb8(ref mut p) => p.blend_pixel(x, y, pixel.to_rgb()),
//             DynamicImage::ImageRgba8(ref mut p) => p.blend_pixel(x, y,
// pixel),             DynamicImage::ImageLuma16(ref mut p) => {
//                 p.blend_pixel(x, y, pixel.to_luma().into_color())
//             }
//             DynamicImage::ImageLumaA16(ref mut p) => {
//                 p.blend_pixel(x, y, pixel.to_luma_alpha().into_color())
//             }
//             DynamicImage::ImageRgb16(ref mut p) => p.blend_pixel(x, y,
// pixel.to_rgb().into_color()),             DynamicImage::ImageRgba16(ref mut
// p) => p.blend_pixel(x, y, pixel.into_color()),
// DynamicImage::ImageRgb32F(ref mut p) => {                 p.blend_pixel(x, y,
// pixel.to_rgb().into_color())             }
//             DynamicImage::ImageRgba32F(ref mut p) => p.blend_pixel(x, y,
// pixel.into_color()),         }
//     }

//     /// Do not use is function: It is unimplemented!
//     fn get_pixel_mut(&mut self, _: u32, _: u32) -> &mut color::Rgba<u8> {
//         unimplemented!()
//     }
// }

impl Default for DynImage {
    fn default() -> Self {
        Self::ImageRgba8(ImageBuffer::default())
    }
}

/// Decodes an image and stores it into a dynamic image
fn decoder_to_image<I: ImageDecoder>(decoder: I) -> ImageResult<DynImage> {
    let (w, h) = decoder.dimensions();
    let color_type = decoder.color_type();

    let image = match color_type {
        image::ColorType::Rgb8 => {
            let buf = decoder_to_vec(decoder)?;
            ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageRgb8)
        }

        image::ColorType::Rgba8 => {
            let buf = decoder_to_vec(decoder)?;
            ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageRgba8)
        }

        image::ColorType::L8 => {
            let buf = decoder_to_vec(decoder)?;
            ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageLuma8)
        }

        image::ColorType::La8 => {
            let buf = decoder_to_vec(decoder)?;
            ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageLumaA8)
        }

        image::ColorType::Rgb16 => {
            let buf = decoder_to_vec(decoder)?;
            ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageRgb16)
        }

        image::ColorType::Rgba16 => {
            let buf = decoder_to_vec(decoder)?;
            ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageRgba16)
        }

        image::ColorType::Rgb32F => {
            let buf = decoder_to_vec(decoder)?;
            ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageRgb32F)
        }

        image::ColorType::Rgba32F => {
            let buf = decoder_to_vec(decoder)?;
            ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageRgba32F)
        }

        image::ColorType::L16 => {
            let buf = decoder_to_vec(decoder)?;
            ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageLuma16)
        }

        image::ColorType::La16 => {
            let buf = decoder_to_vec(decoder)?;
            ImageBuffer::from_raw(w, h, buf).map(DynImage::ImageLumaA16)
        }
        _ => panic!("Unkown Colortype"),
    };

    match image {
        Some(image) => Ok(image),
        None => Err(ImageError::Parameter(ParameterError::from_kind(ParameterErrorKind::DimensionMismatch))),
    }
}

/// Open the image located at the path specified.
/// The image's format is determined from the path's file extension.
///
/// Try [`io::Reader`] for more advanced uses, including guessing the format
/// based on the file's content before its path.
///
/// [`io::Reader`]: io/struct.Reader.html
#[allow(clippy::missing_errors_doc)]
pub fn open<P>(path: P) -> ImageResult<DynImage>
where
    P: AsRef<Path>,
{
    image::ImageReader::open(path)?.decode().map(|v| v.try_into().unwrap())
}

/// Read a tuple containing the (width, height) of the image located at the
/// specified path. This is faster than fully loading the image and then getting
/// its dimensions.
///
/// Try [`io::Reader`] for more advanced uses, including guessing the format
/// based on the file's content before its path or manually supplying the
/// format.
///
/// [`io::Reader`]: io/struct.Reader.html
#[allow(clippy::missing_errors_doc)]
pub fn image_dimensions<P>(path: P) -> ImageResult<(u32, u32)>
where
    P: AsRef<Path>,
{
    image::ImageReader::open(path)?.into_dimensions()
}

/// Saves the supplied buffer to a file at the path specified.
///
/// The image format is derived from the file extension. The buffer is assumed
/// to have the correct format according to the specified color type.
///
/// This will lead to corrupted files if the buffer contains malformed data.
/// Currently only jpeg, png, ico, pnm, bmp, exr and tiff files are supported.
#[allow(clippy::missing_errors_doc)]
pub fn save_buffer(path: impl AsRef<Path>, buf: &[u8], width: u32, height: u32, color: image::ColorType) -> ImageResult<()> {
    // thin wrapper function to strip generics before calling save_buffer_impl
    free_functions::save_buffer_impl(path.as_ref(), buf, width, height, color)
}

/// Saves the supplied buffer to a file at the path specified
/// in the specified format.
///
/// The buffer is assumed to have the correct format according
/// to the specified color type.
/// This will lead to corrupted files if the buffer contains
/// malformed data. Currently only jpeg, png, ico, bmp, exr and
/// tiff files are supported.
#[allow(clippy::missing_errors_doc)]
pub fn save_buffer_with_format(
    path: impl AsRef<Path>,
    buf: &[u8],
    width: u32,
    height: u32,
    color: image::ColorType,
    format: ImageFormat,
) -> ImageResult<()> {
    // thin wrapper function to strip generics
    free_functions::save_buffer_with_format_impl(path.as_ref(), buf, width, height, color, format)
}

/// Writes the supplied buffer to a writer in the specified format.
///
/// The buffer is assumed to have the correct format according to the specified
/// color type. This will lead to corrupted writers if the buffer contains
/// malformed data.
///
/// Assumes the writer is buffered. In most cases, you should wrap your writer
/// in a `BufWriter` for best performance.
#[allow(clippy::missing_errors_doc)]
pub fn write_buffer_with_format<W: Write + Seek>(
    buffered_writer: &mut W,
    buf: &[u8],
    width: u32,
    height: u32,
    color: image::ColorType,
    format: ImageFormat,
) -> ImageResult<()> {
    // thin wrapper function to strip generics
    free_functions::write_buffer_impl(buffered_writer, buf, width, height, color, format)
}

/// Create a new image from a byte slice
///
/// Makes an educated guess about the image format.
/// TGA is not supported by this function.
///
/// Try [`io::Reader`] for more advanced uses.
///
/// [`io::Reader`]: io/struct.Reader.html
#[allow(clippy::missing_errors_doc)]
pub fn load_from_memory(buffer: &[u8]) -> ImageResult<DynImage> {
    let format = free_functions::guess_format(buffer)?;
    load_from_memory_with_format(buffer, format)
}

/// Create a new image from a byte slice
///
/// This is just a simple wrapper that constructs an `std::io::Cursor` around
/// the buffer and then calls `load` with that reader.
///
/// Try [`io::Reader`] for more advanced uses.
///
/// [`load`]: fn.load.html
/// [`io::Reader`]: io/struct.Reader.html
#[inline(always)]
#[allow(clippy::missing_errors_doc)]
#[allow(clippy::inline_always)]
pub fn load_from_memory_with_format(buf: &[u8], format: ImageFormat) -> ImageResult<DynImage> {
    let b = io::Cursor::new(buf);
    free_functions::load(b, format).map(|v| v.try_into().unwrap())
}
