//! One way to the contents of an entity, as you might do for debugging. A
//! similar pattern could also be useful for serialization, or other
//! row-oriented generic operations.

#[allow(clippy::type_complexity)]
fn format_entity(entity: gloss_hecs::EntityRef<'_>) -> String {
    fn fmt<T: gloss_hecs::Component + std::fmt::Display>(entity: gloss_hecs::EntityRef<'_>) -> Option<String> {
        Some(entity.get::<&T>()?.to_string())
    }

    const FUNCTIONS: &[&dyn Fn(gloss_hecs::EntityRef<'_>) -> Option<String>] = &[&fmt::<i32>, &fmt::<bool>, &fmt::<f64>];

    let mut out = String::new();
    for f in FUNCTIONS {
        if let Some(x) = f(entity) {
            if out.is_empty() {
                out.push('[');
            } else {
                out.push_str(", ");
            }
            out.push_str(&x);
        }
    }
    if out.is_empty() {
        out.push_str("[]");
    } else {
        out.push(']');
    }
    out
}

fn main() {
    let mut world = gloss_hecs::World::new();
    let e = world.spawn((42, true));
    println!("{}", format_entity(world.entity(e).unwrap()));
}
