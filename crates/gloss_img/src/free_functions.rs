//directly from image crate because I need them and they are all private

use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Seek},
    path::Path,
};

// use image::ImageFormat;
// use crate::color::ExtendedColorType;

#[allow(clippy::wildcard_imports)]
use image::codecs::*;
use image::{
    error::{ImageError, ImageFormatHint, ImageResult},
    DynamicImage, ImageFormat,
};
#[allow(unused_imports)] // When no features are supported
use image::{ImageDecoder, ImageEncoder};

/// Create a new image from a Reader.
///
/// Assumes the reader is already buffered. For optimal performance,
/// consider wrapping the reader with a `BufReader::new()`.
///
/// Try [`io::Reader`] for more advanced uses.
///
/// [`io::Reader`]: io/struct.Reader.html
#[allow(unused_variables)]
// r is unused if no features are supported.
#[allow(clippy::missing_errors_doc)]
pub fn load<R: BufRead + Seek>(r: R, format: ImageFormat) -> ImageResult<DynamicImage> {
    load_inner(r, image::Limits::default(), format)
}

pub(crate) trait DecoderVisitor {
    type Result;
    fn visit_decoder<D: ImageDecoder>(self, decoder: D) -> ImageResult<Self::Result>;
}

pub(crate) fn load_decoder<R: BufRead + Seek, V: DecoderVisitor>(
    r: R,
    format: ImageFormat,
    limits: image::Limits,
    visitor: V,
) -> ImageResult<V::Result> {
    #[allow(unreachable_patterns)]
    // Default is unreachable if all features are supported.
    match format {
        // #[cfg(feature = "avif-decoder")]
        // image::ImageFormat::Avif => visitor.visit_decoder(avif::AvifDecoder::new(r)?),
        #[cfg(feature = "png")]
        image::ImageFormat::Png => visitor.visit_decoder(png::PngDecoder::with_limits(r, limits)?),
        #[cfg(feature = "gif")]
        image::ImageFormat::Gif => visitor.visit_decoder(gif::GifDecoder::new(r)?),
        #[cfg(feature = "jpeg")]
        image::ImageFormat::Jpeg => visitor.visit_decoder(jpeg::JpegDecoder::new(r)?),
        #[cfg(feature = "webp")]
        image::ImageFormat::WebP => visitor.visit_decoder(webp::WebPDecoder::new(r)?),
        #[cfg(feature = "tiff")]
        image::ImageFormat::Tiff => visitor.visit_decoder(tiff::TiffDecoder::new(r)?),
        #[cfg(feature = "tga")]
        image::ImageFormat::Tga => visitor.visit_decoder(tga::TgaDecoder::new(r)?),
        #[cfg(feature = "bmp")]
        image::ImageFormat::Bmp => visitor.visit_decoder(bmp::BmpDecoder::new(r)?),
        #[cfg(feature = "ico")]
        image::ImageFormat::Ico => visitor.visit_decoder(ico::IcoDecoder::new(r)?),
        #[cfg(feature = "hdr")]
        image::ImageFormat::Hdr => visitor.visit_decoder(hdr::HdrDecoder::new(r)?),
        #[cfg(feature = "openexr")]
        image::ImageFormat::OpenExr => visitor.visit_decoder(openexr::OpenExrDecoder::new(r)?),
        #[cfg(feature = "pnm")]
        image::ImageFormat::Pnm => visitor.visit_decoder(pnm::PnmDecoder::new(r)?),
        #[cfg(feature = "qoi")]
        image::ImageFormat::Qoi => visitor.visit_decoder(qoi::QoiDecoder::new(r)?),
        _ => Err(ImageError::Unsupported(ImageFormatHint::Exact(format).into())),
    }
}

pub(crate) fn load_inner<R: BufRead + Seek>(r: R, limits: image::Limits, format: ImageFormat) -> ImageResult<DynamicImage> {
    struct LoadVisitor(image::Limits);
    impl DecoderVisitor for LoadVisitor {
        type Result = DynamicImage;

        fn visit_decoder<'a, D: ImageDecoder>(self, mut decoder: D) -> ImageResult<Self::Result> {
            let mut limits = self.0;
            // Check that we do not allocate a bigger buffer than we are allowed to
            // FIXME: should this rather go in `DynamicImage::from_decoder` somehow?
            limits.reserve(decoder.total_bytes())?;
            decoder.set_limits(limits)?;
            DynamicImage::from_decoder(decoder)
        }
    }

    load_decoder(r, format, limits.clone(), LoadVisitor(limits))
}

#[allow(clippy::missing_errors_doc)]
pub fn image_dimensions_impl(path: &Path) -> ImageResult<(u32, u32)> {
    let format = image::ImageFormat::from_path(path)?;
    let reader = BufReader::new(File::open(path)?);
    image_dimensions_with_format_impl(reader, format)
}

#[allow(unused_variables)]
#[allow(clippy::missing_errors_doc)]
// fin is unused if no features are supported.
pub fn image_dimensions_with_format_impl<R: BufRead + Seek>(buffered_read: R, format: ImageFormat) -> ImageResult<(u32, u32)> {
    struct DimVisitor;

    impl DecoderVisitor for DimVisitor {
        type Result = (u32, u32);
        fn visit_decoder<'a, D: ImageDecoder>(self, decoder: D) -> ImageResult<Self::Result> {
            Ok(decoder.dimensions())
        }
    }

    load_decoder(buffered_read, format, image::Limits::default(), DimVisitor)
}

#[allow(unused_variables)]
// Most variables when no features are supported
pub(crate) fn save_buffer_impl(path: &Path, buf: &[u8], width: u32, height: u32, color: image::ColorType) -> ImageResult<()> {
    let format = ImageFormat::from_path(path)?;
    save_buffer_with_format_impl(path, buf, width, height, color, format)
}

#[allow(unused_variables)]
// Most variables when no features are supported
pub(crate) fn save_buffer_with_format_impl(
    path: &Path,
    buf: &[u8],
    width: u32,
    height: u32,
    color: image::ColorType,
    format: ImageFormat,
) -> ImageResult<()> {
    let buffered_file_write = &mut BufWriter::new(File::create(path)?); // always seekable

    let format = match format {
        #[cfg(feature = "pnm")]
        image::ImageFormat::Pnm => {
            let ext = path.extension().and_then(|s| s.to_str()).map_or(String::new(), str::to_ascii_lowercase);
            match &*ext {
                "pbm" => pnm::PnmSubtype::Bitmap(pnm::SampleEncoding::Binary),
                "pgm" => pnm::PnmSubtype::Graymap(pnm::SampleEncoding::Binary),
                "ppm" => pnm::PnmSubtype::Pixmap(pnm::SampleEncoding::Binary),
                "pam" => pnm::PnmSubtype::ArbitraryMap,
                _ => return Err(ImageError::Unsupported(ImageFormatHint::Exact(format).into())), // Unsupported Pnm subtype.
            };
            image::ImageFormat::Pnm // Return Pnm directly
        }
        format => format,
    };

    write_buffer_impl(buffered_file_write, buf, width, height, color, format)
}

#[allow(unused_variables)]
// Most variables when no features are supported
pub(crate) fn write_buffer_impl<W: std::io::Write + Seek>(
    buffered_write: &mut W,
    buf: &[u8],
    width: u32,
    height: u32,
    color: image::ColorType,
    format: ImageFormat,
) -> ImageResult<()> {
    match format {
        #[cfg(feature = "png")]
        ImageFormat::Png => png::PngEncoder::new(buffered_write).write_image(buf, width, height, color.into()),
        #[cfg(feature = "jpeg")]
        // TODO: Jpeg does have quality anymore
        ImageFormat::Jpeg => jpeg::JpegEncoder::new_with_quality(buffered_write, 100).write_image(buf, width, height, color.into()),
        #[cfg(feature = "pnm")]
        ImageFormat::Pnm => pnm::PnmEncoder::new(buffered_write).write_image(buf, width, height, color.into()),
        #[cfg(feature = "gif")]
        ImageFormat::Gif => gif::GifEncoder::new(buffered_write).encode(buf, width, height, color.into()),
        #[cfg(feature = "ico")]
        ImageFormat::Ico => ico::IcoEncoder::new(buffered_write).write_image(buf, width, height, color.into()),
        #[cfg(feature = "bmp")]
        ImageFormat::Bmp => bmp::BmpEncoder::new(buffered_write).write_image(buf, width, height, color.into()),
        // #[cfg(feature = "farbfeld")]
        // ImageFormat::Farbfeld => farbfeld::FarbfeldEncoder::new(buffered_write).write_image(buf, width, height, color.into()),
        #[cfg(feature = "tga")]
        ImageFormat::Tga => tga::TgaEncoder::new(buffered_write).write_image(buf, width, height, color.into()),
        #[cfg(feature = "openexr")]
        ImageFormat::OpenExr => openexr::OpenExrEncoder::new(buffered_write).write_image(buf, width, height, color.into()),
        #[cfg(feature = "tiff")]
        ImageFormat::Tiff => tiff::TiffEncoder::new(buffered_write).write_image(buf, width, height, color.into()),
        // #[cfg(feature = "avif-encoder")]
        // ImageFormat::Avif => {
        //     avif::AvifEncoder::new(buffered_write).write_image(buf, width, height, color.into())
        // }
        #[cfg(feature = "qoi")]
        ImageFormat::Qoi => qoi::QoiEncoder::new(buffered_write).write_image(buf, width, height, color.into()),
        #[cfg(feature = "webp")]
        ImageFormat::WebP => webp::WebPEncoder::new_lossless(buffered_write).write_image(buf, width, height, color.into()),

        // image::ImageFormat::Unsupported(msg) => Err(ImageError::Unsupported(UnsupportedError::from_format_and_kind(
        //     ImageFormatHint::Unknown,
        //     UnsupportedErrorKind::Format(ImageFormatHint::Name(msg)),
        // ))),
        // ImageFormat::Unsupported(_) => panic! {"Unkown format"},
        _ => panic! {"Unkown format"},
    }
}

static MAGIC_BYTES: [(&[u8], ImageFormat); 23] = [
    (b"\x89PNG\r\n\x1a\n", ImageFormat::Png),
    (&[0xff, 0xd8, 0xff], ImageFormat::Jpeg),
    (b"GIF89a", ImageFormat::Gif),
    (b"GIF87a", ImageFormat::Gif),
    (b"RIFF", ImageFormat::WebP), // TODO: better magic byte detection, see https://github.com/image-rs/image/issues/660
    (b"MM\x00*", ImageFormat::Tiff),
    (b"II*\x00", ImageFormat::Tiff),
    (b"DDS ", ImageFormat::Dds),
    (b"BM", ImageFormat::Bmp),
    (&[0, 0, 1, 0], ImageFormat::Ico),
    (b"#?RADIANCE", ImageFormat::Hdr),
    (b"P1", ImageFormat::Pnm),
    (b"P2", ImageFormat::Pnm),
    (b"P3", ImageFormat::Pnm),
    (b"P4", ImageFormat::Pnm),
    (b"P5", ImageFormat::Pnm),
    (b"P6", ImageFormat::Pnm),
    (b"P7", ImageFormat::Pnm),
    (b"farbfeld", ImageFormat::Farbfeld),
    (b"\0\0\0 ftypavif", ImageFormat::Avif),
    (b"\0\0\0\x1cftypavif", ImageFormat::Avif),
    (&[0x76, 0x2f, 0x31, 0x01], ImageFormat::OpenExr), // = &exr::meta::magic_number::BYTES
    (b"qoif", ImageFormat::Qoi),
];

/// Guess image format from memory block
///
/// Makes an educated guess about the image format based on the Magic Bytes at
/// the beginning. TGA is not supported by this function.
/// This is not to be trusted on the validity of the whole memory block
#[allow(clippy::missing_errors_doc)]
pub fn guess_format(buffer: &[u8]) -> ImageResult<ImageFormat> {
    match guess_format_impl(buffer) {
        Some(format) => Ok(format),
        None => Err(ImageError::Unsupported(ImageFormatHint::Unknown.into())),
    }
}

pub(crate) fn guess_format_impl(buffer: &[u8]) -> Option<ImageFormat> {
    for &(signature, format) in &MAGIC_BYTES {
        if buffer.starts_with(signature) {
            return Some(format);
        }
    }

    None
}
