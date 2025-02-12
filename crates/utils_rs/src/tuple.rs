// use core::iter::once;

// /// Iterate over tuples
// /// https://stackoverflow.com/a/58349663
// macro_rules! chained_elements {
//     ($exp: expr) => {
//         core::iter::once($exp as _)
//     };
//     ($exp: expr, $($rest:tt)*) => {
//         core::iter::once($exp as _)
//         .chain(chained_elements!($($rest)*))
//     }
// }
// pub fn mut_tuple_to_iter(v: &mut ((), i32)) -> impl Iterator<Item = &mut dyn D> {
//     chained_elements!(&mut v.0, &mut v.1)
// }
