use std::ops::MulAssign;

// use na::SimdPartialOrd;
use nalgebra as na;
// use std::ops::Add;

pub struct AcesFitted {
    rgb_to_rrt: na::Matrix3<f32>,
    odt_to_rgb: na::Matrix3<f32>,
}
impl AcesFitted {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
        let rgb_to_rrt = na::Matrix3::<f32>::from_columns(&[
            na::Vector3::new(0.59719, 0.35458, 0.04823),
            na::Vector3::new(0.07600, 0.90834, 0.01566),
            na::Vector3::new(0.02840, 0.13383, 0.83777),
        ]);
        let odt_to_rgb = na::Matrix3::<f32>::from_columns(&[
            na::Vector3::new(1.60475, -0.53108, -0.07367),
            na::Vector3::new(-0.10208, 1.10813, -0.00605),
            na::Vector3::new(-0.00327, -0.07276, 1.07602),
        ]);
        Self { rgb_to_rrt, odt_to_rgb }
    }
    pub fn tonemap(&self, color: &na::Vector3<f32>) -> na::Vector3<f32> {
        let mut fitted_color = color.transpose();
        fitted_color.mul_assign(&self.rgb_to_rrt);

        // Apply RRT and ODT
        fitted_color = Self::rrt_and_odt_fit(&fitted_color);

        fitted_color.mul_assign(&self.odt_to_rgb);

        // Clamp to [0, 1]
        fitted_color.x = fitted_color.x.clamp(0.0, 1.0);
        fitted_color.y = fitted_color.y.clamp(0.0, 1.0);
        fitted_color.z = fitted_color.z.clamp(0.0, 1.0);

        fitted_color.transpose()
    }
    fn rrt_and_odt_fit(
        v: &na::Matrix<f32, na::Const<1>, na::Const<3>, na::ArrayStorage<f32, 1, 3>>,
    ) -> na::Matrix<f32, na::Const<1>, na::Const<3>, na::ArrayStorage<f32, 1, 3>> {
        let v1 = v.add_scalar(0.024_578_6);
        let v2 = v.component_mul(&v1);
        let a = v2.add_scalar(-0.000_090_537);

        let v1 = 0.983_729 * v;
        let v2 = v1.add_scalar(0.432_951);
        let v3 = v.component_mul(&v2);
        let b = v3.add_scalar(0.238_081);
        a.component_div(&b)
    }
}
