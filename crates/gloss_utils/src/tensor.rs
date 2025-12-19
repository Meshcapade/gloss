use burn::{
    prelude::Backend,
    tensor::{Float, Tensor},
};

/////
///// Some burn utilities
/////
/// Normalise a 2D tensor across dim 1
pub fn normalize_tensor<B: Backend>(tensor: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
    let norm = tensor.clone().powf_scalar(2.0).sum_dim(1).sqrt(); // Compute the L2 norm along the last axis (dim = 1)
    tensor.div(norm) // Divide each vector by its norm
}

/// Cross product of 2 2D Tensors
pub fn cross_product<B: Backend>(
    a: &Tensor<B, 2, Float>, // Tensor of shape [N, 3]
    b: &Tensor<B, 2, Float>, // Tensor of shape [N, 3]
) -> Tensor<B, 2, Float> {
    // Split the input tensors along dimension 1 (the 3 components) using chunk
    let a_chunks = a.clone().chunk(3, 1); // Split tensor `a` into 3 chunks: ax, ay, az
    let b_chunks = b.clone().chunk(3, 1); // Split tensor `b` into 3 chunks: bx, by, bz

    let ax: Tensor<B, 1> = a_chunks[0].clone().squeeze(1); // x component of a
    let ay: Tensor<B, 1> = a_chunks[1].clone().squeeze(1); // y component of a
    let az: Tensor<B, 1> = a_chunks[2].clone().squeeze(1); // z component of a

    let bx: Tensor<B, 1> = b_chunks[0].clone().squeeze(1); // x component of b
    let by: Tensor<B, 1> = b_chunks[1].clone().squeeze(1); // y component of b
    let bz: Tensor<B, 1> = b_chunks[2].clone().squeeze(1); // z component of b

    // Compute the components of the cross product
    let cx = ay.clone().mul(bz.clone()).sub(az.clone().mul(by.clone())); // cx = ay * bz - az * by
    let cy = az.mul(bx.clone()).sub(ax.clone().mul(bz)); // cy = az * bx - ax * bz
    let cz = ax.mul(by).sub(ay.mul(bx)); // cz = ax * by - ay * bx

    // Stack the result to form the resulting [N, 3] tensor
    Tensor::stack(vec![cx, cy, cz], 1) // Concatenate along the second dimension
}
