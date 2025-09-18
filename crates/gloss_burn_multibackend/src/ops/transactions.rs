use burn::tensor::ops::TransactionOps;

use crate::backend::MultiBackend;

//ops
impl TransactionOps<Self> for MultiBackend {}
