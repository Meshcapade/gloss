//https://blog.frost.kiwi/GLSL-noise-and-radial-gradient/
/* Gradient noise from Jorge Jimenez's presentation: */
/* http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare */
// float gradientNoise(in vec2 uv)
// {
// 	return fract(52.9829189 * fract(dot(uv, vec2(0.06711056, 0.00583715))));
// }
fn gradient_noise( uv: vec2<f32> ) -> f32 {
	return fract(52.9829189 * fract(dot(uv, vec2<f32>(0.06711056, 0.00583715))));
}